"""Research Module - Updated with all research agents"""

from .git_research import GitResearchAgent, Repository, SearchQuery, CodeAnalysis
from .repository_analysis import RepositoryAnalysisAgent, CodeQualityReport, ArchitectureAnalysis
from .rl_research import RLResearchAgent, RLFramework, RLResearchResult
from .vision_research import VisionPipelineResearchAgent, VisionFramework, VisionResearchResult
from .integration_planner import IntegrationPlannerAgent, IntegrationModule, IntegrationPlan
from .benchmark_agent import BenchmarkAgent, BenchmarkResult, ModelRanking

__all__ = [
    "GitResearchAgent",
    "Repository",
    "SearchQuery",
    "CodeAnalysis",
    "RepositoryAnalysisAgent",
    "CodeQualityReport",
    "ArchitectureAnalysis",
    "RLResearchAgent",
    "RLFramework",
    "RLResearchResult",
    "VisionPipelineResearchAgent",
    "VisionFramework",
    "VisionResearchResult",
    "IntegrationPlannerAgent",
    "IntegrationModule",
    "IntegrationPlan",
    "BenchmarkAgent",
    "BenchmarkResult",
    "ModelRanking",
]
