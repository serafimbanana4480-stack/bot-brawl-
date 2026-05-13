"""Git Research Agent - Autonomous Git Repository Search & Analysis"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlencode
import hashlib

from ..agents.base import BaseAgent, AgentConfig, AgentMessage, AgentResponse, AgentType
from ..orchestration.event_bus import EventBus, EventType


@dataclass
class Repository:
    id: int
    name: str
    full_name: str
    description: Optional[str]
    html_url: str
    stars: int
    forks: int
    language: Optional[str]
    topics: List[str]
    created_at: str
    updated_at: str
    pushed_at: str
    open_issues: int
    watchers: int
    score: float
    clone_url: str
    default_branch: str
    license: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "html_url": self.html_url,
            "stars": self.stars,
            "forks": self.forks,
            "language": self.language,
            "topics": self.topics,
            "score": self.score,
        }


@dataclass
class SearchQuery:
    query: str
    language: Optional[str] = None
    sort: str = "stars"
    order: str = "desc"
    per_page: int = 30
    page: int = 1


@dataclass
class CodeAnalysis:
    repository: str
    quality_score: float
    architecture_score: float
    documentation_score: float
    test_coverage: float
    technologies: List[str]
    frameworks: List[str]
    patterns: List[str]
    issues: List[str]
    recommendations: List[str]


class GitResearchAgent(BaseAgent):
    def __init__(self, config: AgentConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        
        self.github_token = None
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = 0
        
        self.search_cache: Dict[str, List[Repository]] = {}
        self.analysis_cache: Dict[str, CodeAnalysis] = {}
        
        self.search_queries = [
            "brawl-stars bot",
            "brawl-stars automation",
            "brawl-stars AI",
            "game bot AI",
            "MOBA bot machine learning",
            "computer vision game bot",
            "reinforcement learning game",
            "YOLOv8 game bot",
            "real-time game automation",
            "arena battle bot",
        ]
        
    async def initialize(self) -> bool:
        await super().initialize()
        return True
    
    async def process(self, message: AgentMessage) -> AgentResponse:
        start_time = time.time()
        action = message.content.get("action", "search")
        
        try:
            if action == "search":
                result = await self._search_repositories(message.content)
            elif action == "analyze":
                result = await self._analyze_repository(message.content)
            elif action == "compare":
                result = await self._compare_repositories(message.content)
            elif action == "trending":
                result = await self._get_trending_repos(message.content)
            elif action == "research":
                result = await self._comprehensive_research(message.content)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            return AgentResponse(
                success=True,
                message=message,
                data=result,
                confidence=0.85,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                message=message,
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cached_repositories": len(self.search_cache),
            "analyzed_repos": len(self.analysis_cache),
            "rate_limit_remaining": self.rate_limit_remaining,
            "confidence": 0.9,
        }
    
    async def _search_repositories(self, content: Dict[str, Any]) -> Dict[str, Any]:
        query = content.get("query", "brawl stars bot")
        language = content.get("language")
        max_results = content.get("max_results", 30)
        
        cache_key = f"{query}_{language}_{max_results}"
        if cache_key in self.search_cache:
            return {
                "cached": True,
                "repositories": [r.to_dict() for r in self.search_cache[cache_key][:max_results]],
                "total": len(self.search_cache[cache_key]),
            }
        
        search_queries = [
            f"{query} bot",
            f"{query} AI",
            f"{query} automation",
            f"{query} machine learning",
            f"{query} computer vision",
        ]
        
        all_repos = []
        seen_ids = set()
        
        for q in search_queries:
            repos = await self._search_github(q, language)
            for repo in repos:
                if repo.id not in seen_ids:
                    seen_ids.add(repo.id)
                    all_repos.append(repo)
            
            await asyncio.sleep(0.5)
        
        all_repos.sort(key=lambda r: (r.stars + r.forks * 0.5), reverse=True)
        
        self.search_cache[cache_key] = all_repos
        
        await self.event_bus.publish(Event(
            source=self.id,
            type=EventType.DECISION_PROPOSED,
            data={
                "action": "search_completed",
                "query": query,
                "found": len(all_repos),
            },
        ))
        
        return {
            "cached": False,
            "query": query,
            "repositories": [r.to_dict() for r in all_repos[:max_results]],
            "total_found": len(all_repos),
            "search_queries_used": search_queries,
        }
    
    async def _search_github(self, query: str, language: Optional[str] = None,
                           per_page: int = 30) -> List[Repository]:
        if time.time() < self.rate_limit_reset:
            await asyncio.sleep(self.rate_limit_reset - time.time() + 1)
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Enterprise-AI-Platform",
        }
        
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(per_page, 100),
        }
        
        if language:
            params["q"] += f" language:{language}"
        
        url = f"https://api.github.com/search/repositories?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 5000))
                        self.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))
                        
                        data = await response.json()
                        repos = []
                        
                        for item in data.get("items", []):
                            repo = Repository(
                                id=item["id"],
                                name=item["name"],
                                full_name=item["full_name"],
                                description=item.get("description"),
                                html_url=item["html_url"],
                                stars=item["stargazers_count"],
                                forks=item["forks_count"],
                                language=item.get("language"),
                                topics=item.get("topics", []),
                                created_at=item["created_at"],
                                updated_at=item["updated_at"],
                                pushed_at=item["pushed_at"],
                                open_issues=item["open_issues_count"],
                                watchers=item["watchers_count"],
                                score=item.get("score", 0),
                                clone_url=item["clone_url"],
                                default_branch=item.get("default_branch", "main"),
                                license=item.get("license", {}).get("name") if item.get("license") else None,
                            )
                            repos.append(repo)
                        
                        return repos
                    elif response.status == 403:
                        self.rate_limit_remaining = 0
                        self.rate_limit_reset = int(time.time()) + 3600
                        return []
                    else:
                        return []
        except Exception as e:
            self.logger.error(f"GitHub search error: {e}")
            return []
    
    async def _analyze_repository(self, content: Dict[str, Any]) -> Dict[str, Any]:
        repo_url = content.get("repo_url")
        if not repo_url:
            return {"error": "No repository URL provided"}
        
        repo_id = self._extract_repo_id(repo_url)
        
        if repo_id in self.analysis_cache:
            return {
                "cached": True,
                "analysis": self.analysis_cache[repo_id].__dict__,
            }
        
        repo_info = await self._get_repo_info(repo_id)
        if not repo_info:
            return {"error": "Repository not found"}
        
        readme = await self._get_readme(repo_id)
        contents = await self._get_contents(repo_id)
        
        analysis = await self._perform_analysis(repo_id, repo_info, readme, contents)
        
        self.analysis_cache[repo_id] = analysis
        
        return {
            "cached": False,
            "analysis": analysis.__dict__,
        }
    
    async def _compare_repositories(self, content: Dict[str, Any]) -> Dict[str, Any]:
        repo_urls = content.get("repositories", [])
        
        comparisons = []
        for url in repo_urls:
            repo_id = self._extract_repo_id(url)
            repo_info = await self._get_repo_info(repo_id)
            if repo_info:
                comparisons.append({
                    "repo_id": repo_id,
                    "name": repo_info.get("name"),
                    "stars": repo_info.get("stargazers_count", 0),
                    "forks": repo_info.get("forks_count", 0),
                    "language": repo_info.get("language"),
                    "description": repo_info.get("description"),
                })
        
        comparisons.sort(key=lambda x: x["stars"], reverse=True)
        
        return {
            "comparisons": comparisons,
            "best_repository": comparisons[0] if comparisons else None,
            "recommendation": self._generate_recommendation(comparisons),
        }
    
    async def _get_trending_repos(self, content: Dict[str, Any]) -> Dict[str, Any]:
        language = content.get("language", "Python")
        since = content.get("since", "daily")
        
        trending_url = f"https://api.github.com/search/repositories"
        
        params = {
            "q": f"created:>{self._get_date_since(since)}",
            "sort": "stars",
            "order": "desc",
            "per_page": 30,
        }
        
        if language:
            params["q"] += f" language:{language}"
        
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        url = f"https://api.github.com/search/repositories?{urlencode(params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        repos = [
                            {
                                "name": item["name"],
                                "full_name": item["full_name"],
                                "stars": item["stargazers_count"],
                                "description": item.get("description"),
                                "language": item.get("language"),
                                "url": item["html_url"],
                            }
                            for item in data.get("items", [])[:20]
                        ]
                        return {"trending": repos, "language": language, "since": since}
        except Exception as e:
            return {"error": str(e)}
        
        return {"trending": [], "language": language, "since": since}
    
    async def _comprehensive_research(self, content: Dict[str, Any]) -> Dict[str, Any]:
        topics = content.get("topics", [
            "brawl-stars",
            "game-bot",
            "computer-vision",
            "reinforcement-learning",
            "yolo",
            "game-ai",
        ])
        
        all_repos = []
        topic_repos = {}
        
        for topic in topics:
            repos = await self._search_github(f"topic:{topic}", per_page=20)
            topic_repos[topic] = [r.to_dict() for r in repos[:5]]
            all_repos.extend(repos)
        
        all_repos.sort(key=lambda r: r.stars, reverse=True)
        
        top_repos = []
        for repo in all_repos[:10]:
            analysis = await self._analyze_repository({
                "repo_url": repo.html_url,
            })
            top_repos.append({
                "repository": repo.to_dict(),
                "analysis": analysis.get("analysis"),
            })
        
        return {
            "topics_searched": topics,
            "total_repos_found": len(all_repos),
            "top_repositories": top_repos,
            "topic_breakdown": topic_repos,
            "recommendations": self._generate_research_recommendations(top_repos),
        }
    
    async def _get_repo_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        url = f"https://api.github.com/repos/{repo_id}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception:
            return None
        
        return None
    
    async def _get_readme(self, repo_id: str) -> Optional[str]:
        url = f"https://api.github.com/repos/{repo_id}/readme"
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        import base64
                        content = data.get("content", "")
                        return base64.b64decode(content).decode("utf-8", errors="ignore")
        except Exception:
            return None
        
        return None
    
    async def _get_contents(self, repo_id: str, path: str = "") -> List[Dict[str, Any]]:
        url = f"https://api.github.com/repos/{repo_id}/contents/{path}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception:
            return []
        
        return []
    
    async def _perform_analysis(self, repo_id: str, repo_info: Dict,
                             readme: Optional[str], contents: List[Dict]) -> CodeAnalysis:
        quality_score = 0.5
        architecture_score = 0.5
        documentation_score = 0.5
        
        if readme:
            doc_length = len(readme)
            documentation_score = min(1.0, doc_length / 5000)
        
        technologies = []
        frameworks = []
        patterns = []
        issues = []
        recommendations = []
        
        languages = set()
        
        for content in contents[:20]:
            if content.get("type") == "file":
                name = content.get("name", "")
                if name.endswith(".py"):
                    languages.add("Python")
                    if "model" in name.lower():
                        frameworks.append("ML Model")
                    if "train" in name.lower():
                        frameworks.append("Training")
                elif name.endswith(".js") or name.endswith(".ts"):
                    languages.add("JavaScript/TypeScript")
                elif name.endswith(".yml") or name.endswith(".yaml"):
                    frameworks.append("CI/CD")
        
        if repo_info.get("language"):
            languages.add(repo_info["language"])
        
        stars = repo_info.get("stargazers_count", 0)
        if stars > 1000:
            quality_score += 0.2
        elif stars > 100:
            quality_score += 0.1
        
        if repo_info.get("forks_count", 0) > 50:
            architecture_score += 0.1
        
        if "bot" in repo_info.get("name", "").lower():
            patterns.append("Game Bot Architecture")
        if any(t in repo_info.get("topics", []) for t in ["machine-learning", "deep-learning"]):
            frameworks.append("Deep Learning")
            patterns.append("ML Pipeline")
        if any(t in repo_info.get("topics", []) for t in ["computer-vision", "yolo", "opencv"]):
            frameworks.append("Computer Vision")
            patterns.append("CV Pipeline")
        
        return CodeAnalysis(
            repository=repo_id,
            quality_score=quality_score,
            architecture_score=architecture_score,
            documentation_score=documentation_score,
            test_coverage=0.3,
            technologies=list(languages),
            frameworks=frameworks,
            patterns=patterns,
            issues=issues,
            recommendations=recommendations,
        )
    
    def _extract_repo_id(self, url: str) -> str:
        url = url.replace("https://github.com/", "").replace("http://github.com/", "")
        if url.endswith("/"):
            url = url[:-1]
        return url
    
    def _get_date_since(self, since: str) -> str:
        from datetime import timedelta
        
        days = {"daily": 1, "weekly": 7, "monthly": 30}.get(since, 1)
        date = datetime.now() - timedelta(days=days)
        return date.strftime("%Y-%m-%d")
    
    def _generate_recommendation(self, comparisons: List[Dict]) -> Dict[str, Any]:
        if not comparisons:
            return {"action": "none", "reason": "No repositories to compare"}
        
        best = comparisons[0]
        
        return {
            "action": "integrate",
            "reason": f"{best['name']} has the highest engagement with {best['stars']} stars",
            "repository": best["full_name"],
            "priority": "high" if best["stars"] > 500 else "medium",
        }
    
    def _generate_research_recommendations(self, top_repos: List[Dict]) -> List[Dict[str, Any]]:
        recommendations = []
        
        for repo in top_repos[:3]:
            repo_data = repo.get("repository", {})
            analysis = repo.get("analysis", {})
            
            recommendations.append({
                "repository": repo_data.get("full_name"),
                "url": repo_data.get("html_url"),
                "stars": repo_data.get("stars"),
                "action": "analyze_and_integrate",
                "priority": "high" if repo_data.get("stars", 0) > 100 else "medium",
                "frameworks_found": analysis.get("frameworks", []),
                "patterns_found": analysis.get("patterns", []),
            })
        
        return recommendations
