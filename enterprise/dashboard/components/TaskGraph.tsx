"use client";

import { useEffect, useState } from "react";
import { Network } from "lucide-react";

interface TaskNode {
  id: string;
  label: string;
  status: string;
  type: string;
}

interface TaskEdge {
  from: string;
  to: string;
}

export function TaskGraph() {
  const [graphData, setGraphData] = useState<{
    nodes: TaskNode[];
    edges: TaskEdge[];
  }>({
    nodes: [],
    edges: [],
  });

  useEffect(() => {
    fetch("/api/workflow/graph")
      .then((res) => res.json())
      .then((data) => {
        if (data.nodes) {
          setGraphData(data);
        }
      })
      .catch(() => {
        setGraphData({
          nodes: [
            { id: "1", label: "Supervisor", status: "completed", type: "supervisor" },
            { id: "2", label: "Strategy", status: "running", type: "strategy" },
            { id: "3", label: "Combat", status: "pending", type: "combat" },
            { id: "4", label: "Vision", status: "running", type: "vision" },
            { id: "5", label: "Navigation", status: "pending", type: "navigation" },
          ],
          edges: [
            { from: "1", to: "2" },
            { from: "1", to: "3" },
            { from: "2", to: "4" },
            { from: "2", to: "5" },
          ],
        });
      });
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-500";
      case "running":
        return "bg-blue-500";
      case "pending":
        return "bg-yellow-500";
      case "failed":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  return (
    <div className="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-6">
      <h2 className="text-lg font-semibold flex items-center gap-2 mb-6">
        <Network className="w-5 h-5 text-purple-500" />
        Task Workflow
      </h2>

      <div className="relative h-64 bg-secondary/30 rounded-lg overflow-hidden">
        <svg className="w-full h-full" viewBox="0 0 400 250">
          {graphData.edges.map((edge, i) => {
            const fromNode = graphData.nodes.find((n) => n.id === edge.from);
            const toNode = graphData.nodes.find((n) => n.id === edge.to);
            if (!fromNode || !toNode) return null;

            const x1 = 50 + parseInt(fromNode.id) * 60;
            const y1 = 125;
            const x2 = 50 + parseInt(toNode.id) * 60;
            const y2 = 125;

            return (
              <line
                key={i}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="var(--border)"
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
              />
            );
          })}

          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3.5, 0 7"
                fill="var(--border)"
              />
            </marker>
          </defs>

          {graphData.nodes.map((node) => {
            const x = 50 + parseInt(node.id) * 60;
            const y = 125;

            return (
              <g key={node.id} transform={`translate(${x}, ${y})`}>
                <circle
                  r="24"
                  fill="var(--card)"
                  stroke="var(--border)"
                  strokeWidth="2"
                  className="transition-all hover:stroke-purple-500"
                />
                <circle
                  r="6"
                  cx="0"
                  cy="-20"
                  className={getStatusColor(node.status)}
                />
                <text
                  textAnchor="middle"
                  dy="4"
                  className="text-xs fill-foreground font-medium"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>

        <div className="absolute bottom-4 left-4 flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="text-muted-foreground">Completed</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span className="text-muted-foreground">Running</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <span className="text-muted-foreground">Pending</span>
          </div>
        </div>
      </div>
    </div>
  );
}
