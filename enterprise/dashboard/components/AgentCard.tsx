"use client";

import { useState } from "react";
import { Brain, Crosshair, Eye, Map, Target, Trophy, Lightbulb, MemoryStick, Shield, Users } from "lucide-react";

interface Agent {
  id: string;
  name: string;
  type: string;
  status: string;
  metrics: any;
}

interface AgentCardProps {
  agents: Agent[];
  selectedAgent: string | null;
  onSelectAgent: (id: string) => void;
}

const agentIcons: Record<string, any> = {
  supervisor: Users,
  strategy: Target,
  combat: Crosshair,
  vision: Eye,
  navigation: Map,
  tactical: Target,
  replay: Trophy,
  learning: Brain,
  memory: MemoryStick,
  reflection: Lightbulb,
  coordination: Shield,
};

const agentColors: Record<string, string> = {
  supervisor: "from-red-500 to-orange-500",
  strategy: "from-blue-500 to-cyan-500",
  combat: "from-red-500 to-rose-500",
  vision: "from-purple-500 to-pink-500",
  navigation: "from-green-500 to-emerald-500",
  tactical: "from-yellow-500 to-amber-500",
  replay: "from-indigo-500 to-blue-500",
  learning: "from-violet-500 to-purple-500",
  memory: "from-teal-500 to-cyan-500",
  reflection: "from-orange-500 to-yellow-500",
  coordination: "from-pink-500 to-rose-500",
};

export function AgentCard({ agents, selectedAgent, onSelectAgent }: AgentCardProps) {
  const [filter, setFilter] = useState<string>("all");

  const filteredAgents = filter === "all"
    ? agents
    : agents.filter((a) => a.type === filter);

  return (
    <div className="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-500" />
          Active Agents
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => setFilter("all")}
            className={cn(
              "px-3 py-1 text-xs rounded-full transition-colors",
              filter === "all"
                ? "bg-purple-500 text-white"
                : "bg-secondary text-muted-foreground hover:bg-secondary/80"
            )}
          >
            All ({agents.length})
          </button>
          <button
            onClick={() => setFilter("active")}
            className={cn(
              "px-3 py-1 text-xs rounded-full transition-colors",
              filter === "active"
                ? "bg-green-500 text-white"
                : "bg-secondary text-muted-foreground hover:bg-secondary/80"
            )}
          >
            Active
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredAgents.map((agent) => {
          const Icon = agentIcons[agent.type] || Brain;
          const colorClass = agentColors[agent.type] || "from-gray-500 to-gray-600";
          const isActive = agent.status === "processing" || agent.status === "idle";

          return (
            <button
              key={agent.id}
              onClick={() => onSelectAgent(agent.id)}
              className={cn(
                "relative overflow-hidden rounded-lg border p-4 text-left transition-all hover:scale-[1.02]",
                selectedAgent === agent.id
                  ? "border-purple-500 ring-2 ring-purple-500/20"
                  : "border-border hover:border-purple-500/50"
              )}
            >
              <div
                className={cn(
                  "absolute inset-0 bg-gradient-to-br opacity-10",
                  colorClass
                )}
              />

              <div className="relative">
                <div className="flex items-center gap-3 mb-3">
                  <div
                    className={cn(
                      "w-10 h-10 rounded-lg bg-gradient-to-br flex items-center justify-center",
                      colorClass
                    )}
                  >
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-medium capitalize">{agent.name}</h3>
                    <p className="text-xs text-muted-foreground">{agent.type}</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">Status</span>
                    <span
                      className={cn(
                        "px-2 py-0.5 text-xs rounded-full",
                        isActive
                          ? "bg-green-500/20 text-green-500"
                          : "bg-yellow-500/20 text-yellow-500"
                      )}
                    >
                      {agent.status}
                    </span>
                  </div>

                  {agent.metrics && (
                    <>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">Messages</span>
                        <span className="text-xs">
                          {agent.metrics.messages_received || 0}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">Confidence</span>
                        <span className="text-xs">
                          {((agent.metrics.average_confidence || 0.8) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </>
                  )}
                </div>

                <div className="mt-3 h-1 rounded-full bg-secondary overflow-hidden">
                  <div
                    className={cn("h-full bg-gradient-to-r", colorClass)}
                    style={{
                      width: `${((agent.metrics?.average_confidence || 0.8) * 100)}%`,
                    }}
                  />
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {filteredAgents.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No agents found</p>
        </div>
      )}
    </div>
  );
}
