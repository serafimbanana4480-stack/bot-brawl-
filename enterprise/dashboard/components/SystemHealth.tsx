"use client";

import { useState, useEffect } from "react";
import { Wifi, WifiOff, AlertTriangle, CheckCircle } from "lucide-react";

interface SystemHealthProps {
  status: any;
}

export function SystemHealth({ status }: SystemHealthProps) {
  const [health, setHealth] = useState({
    overall: "healthy",
    components: [
      { name: "API Server", status: "healthy", latency: 12 },
      { name: "Event Bus", status: "healthy", latency: 3 },
      { name: "Memory System", status: "healthy", latency: 5 },
      { name: "Agent Pool", status: "healthy", latency: 8 },
    ],
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case "degraded":
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case "down":
        return <WifiOff className="w-4 h-4 text-red-500" />;
      default:
        return <Wifi className="w-4 h-4 text-muted-foreground" />;
    }
  };

  return (
    <div className="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-6">
      <h2 className="text-lg font-semibold flex items-center gap-2 mb-6">
        {health.overall === "healthy" ? (
          <CheckCircle className="w-5 h-5 text-green-500" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-yellow-500" />
        )}
        System Health
      </h2>

      <div className="space-y-3">
        {health.components.map((component) => (
          <div
            key={component.name}
            className="flex items-center justify-between p-3 rounded-lg bg-secondary/50"
          >
            <div className="flex items-center gap-3">
              {getStatusIcon(component.status)}
              <span className="text-sm font-medium">{component.name}</span>
            </div>
            <span className="text-xs text-muted-foreground">
              {component.latency}ms
            </span>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-sm font-medium text-green-500">
            All Systems Operational
          </span>
        </div>
      </div>
    </div>
  );
}
