"""Comprehensive GitHub Research for Brawl Stars and Game Bot Technologies"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List
from enterprise.orchestration.event_bus import EventBus
from enterprise.research.git_research import GitResearchAgent
from enterprise.research.repository_analysis import RepositoryAnalysisAgent
from enterprise.research.rl_research import RLResearchAgent
from enterprise.research.vision_research import VisionPipelineResearchAgent
from enterprise.agents.base import AgentConfig, AgentType


class BrawlStarsResearcher:
    def __init__(self):
        self.event_bus = EventBus()
        self.results = {
            "brawl_stars_bots": [],
            "game_bots": [],
            "rl_frameworks": [],
            "vision_pipelines": [],
            "integrations": [],
        }
        
    async def run_full_research(self):
        print("=" * 60)
        print("BRAWL STARS & GAME BOT RESEARCH SYSTEM")
        print("=" * 60)
        
        await self._research_brawl_stars_bots()
        await self._research_game_ai()
        await self._research_rl_frameworks()
        await self._research_vision_pipelines()
        
        self._generate_integration_report()
        
        return self.results
    
    async def _research_brawl_stars_bots(self):
        print("\n[1/4] Searching for Brawl Stars bots...")
        
        search_queries = [
            "brawl stars bot",
            "brawl-stars automation",
            "brawlstars AI bot",
            "brawl stars auto player",
            "brawl stars game bot",
            "supercell bot",
        ]
        
        async with aiohttp.ClientSession() as session:
            for query in search_queries:
                repos = await self._search_github(session, query)
                for repo in repos[:10]:
                    if repo["full_name"] not in [r["full_name"] for r in self.results["brawl_stars_bots"]]:
                        self.results["brawl_stars_bots"].append(repo)
                await asyncio.sleep(1)
        
        self.results["brawl_stars_bots"].sort(key=lambda x: x.get("stars", 0), reverse=True)
        print(f"   Found {len(self.results['brawl_stars_bots'])} unique repositories")
        
        if self.results["brawl_stars_bots"]:
            top = self.results["brawl_stars_bots"][0]
            print(f"   Top repo: {top['full_name']} ({top.get('stars', 0)} stars)")
    
    async def _research_game_ai(self):
        print("\n[2/4] Searching for Game AI and Bot frameworks...")
        
        search_queries = [
            "game bot machine learning",
            "reinforcement learning game",
            "MOBA AI bot",
            "real-time game automation",
            "arena battle bot AI",
        ]
        
        async with aiohttp.ClientSession() as session:
            for query in search_queries:
                repos = await self._search_github(session, query)
                for repo in repos[:5]:
                    if repo["full_name"] not in [r["full_name"] for r in self.results["game_bots"]]:
                        self.results["game_bots"].append(repo)
                await asyncio.sleep(1)
        
        self.results["game_bots"].sort(key=lambda x: x.get("stars", 0), reverse=True)
        print(f"   Found {len(self.results['game_bots'])} game AI repositories")
    
    async def _research_rl_frameworks(self):
        print("\n[3/4] Researching RL frameworks for game bots...")
        
        rl_keywords = [
            "reinforcement learning game bot",
            "PPO game AI",
            "SAC reinforcement learning",
            "game AI training",
        ]
        
        async with aiohttp.ClientSession() as session:
            for query in rl_keywords:
                repos = await self._search_github(session, query)
                for repo in repos[:5]:
                    if repo["full_name"] not in [r["full_name"] for r in self.results["rl_frameworks"]]:
                        self.results["rl_frameworks"].append(repo)
                await asyncio.sleep(1)
        
        self.results["rl_frameworks"].sort(key=lambda x: x.get("stars", 0), reverse=True)
        print(f"   Found {len(self.results['rl_frameworks'])} RL frameworks")
    
    async def _research_vision_pipelines(self):
        print("\n[4/4] Researching Computer Vision pipelines...")
        
        cv_keywords = [
            "YOLOv8 game detection",
            "computer vision real-time",
            "object tracking game bot",
            "DeepSORT YOLO tracking",
        ]
        
        async with aiohttp.ClientSession() as session:
            for query in cv_keywords:
                repos = await self._search_github(session, query)
                for repo in repos[:5]:
                    if repo["full_name"] not in [r["full_name"] for r in self.results["vision_pipelines"]]:
                        self.results["vision_pipelines"].append(repo)
                await asyncio.sleep(1)
        
        self.results["vision_pipelines"].sort(key=lambda x: x.get("stars", 0), reverse=True)
        print(f"   Found {len(self.results['vision_pipelines'])} vision pipelines")
    
    async def _search_github(self, session: aiohttp.ClientSession, 
                           query: str, per_page: int = 15) -> List[Dict]:
        url = "https://api.github.com/search/repositories"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page}
        headers = {"Accept": "application/vnd.github.v3+json"}
        
        try:
            async with session.get(url, params=params, headers=headers,
                                 timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            "name": item.get("name"),
                            "full_name": item.get("full_name"),
                            "description": item.get("description"),
                            "stars": item.get("stargazers_count", 0),
                            "forks": item.get("forks_count", 0),
                            "language": item.get("language"),
                            "topics": item.get("topics", []),
                            "html_url": item.get("html_url"),
                            "clone_url": item.get("clone_url"),
                        }
                        for item in data.get("items", [])[:per_page]
                    ]
        except Exception as e:
            print(f"   Error searching '{query}': {e}")
        
        return []
    
    def _generate_integration_report(self):
        print("\n" + "=" * 60)
        print("INTEGRATION RECOMMENDATIONS")
        print("=" * 60)
        
        print("\n[Tier 1 - Must Have]")
        tier1 = [r for r in self.results["brawl_stars_bots"] if r.get("stars", 0) > 100]
        for repo in tier1[:5]:
            print(f"  • {repo['full_name']}")
            print(f"    Stars: {repo.get('stars', 0)} | Lang: {repo.get('language', 'N/A')}")
            print(f"    {repo.get('description', 'No description')[:60]}...")
        
        print("\n[Tier 2 - High Value]")
        tier2 = [r for r in self.results["brawl_stars_bots"] 
                 if 20 < r.get("stars", 0) <= 100]
        for repo in tier2[:5]:
            print(f"  • {repo['full_name']} ({repo.get('stars', 0)} stars)")
        
        print("\n[Game AI Frameworks]")
        for repo in self.results["game_bots"][:5]:
            print(f"  • {repo['full_name']} ({repo.get('stars', 0)} stars)")
        
        print("\n[RL Frameworks]")
        for repo in self.results["rl_frameworks"][:5]:
            print(f"  • {repo['full_name']} ({repo.get('stars', 0)} stars)")
        
        print("\n[Vision Pipelines]")
        for repo in self.results["vision_pipelines"][:5]:
            print(f"  • {repo['full_name']} ({repo.get('stars', 0)} stars)")
        
        print("\n" + "=" * 60)
        print("TOP 10 REPOSITORIES FOR INTEGRATION")
        print("=" * 60)
        
        all_repos = (
            self.results["brawl_stars_bots"] +
            self.results["game_bots"] +
            self.results["vision_pipelines"]
        )
        all_repos.sort(key=lambda x: x.get("stars", 0), reverse=True)
        
        for i, repo in enumerate(all_repos[:10], 1):
            print(f"\n{i}. {repo['full_name']}")
            print(f"   ⭐ {repo.get('stars', 0)} | 🍴 {repo.get('forks', 0)} | {repo.get('language', 'N/A')}")
            print(f"   📌 {repo.get('html_url', '')}")
            topics = repo.get("topics", [])
            if topics:
                print(f"   🏷️  {', '.join(topics[:5])}")
        
        self._save_results()
    
    def _save_results(self):
        output_file = "enterprise/research_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {output_file}")


async def main():
    researcher = BrawlStarsResearcher()
    await researcher.run_full_research()


if __name__ == "__main__":
    asyncio.run(main())
