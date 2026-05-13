"use client";

import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Activity, Cpu, MemoryStick, Zap } from "lucide-react";

interface HeaderProps {
  systemStatus: any;
}

export function Header({ systemStatus }: HeaderProps) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const uptime = systemStatus?.uptime
    ? `${Math.floor(systemStatus.uptime / 60)}m ${Math.floor(systemStatus.uptime % 60)}s`
    : "0m 0s";

  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                  Enterprise AI Platform
                </h1>
                <p className="text-xs text-muted-foreground">
                  Multi-Agent Strategic Coordination
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-green-500" />
                <span className="text-muted-foreground">Status:</span>
                <span className="text-green-500 font-medium">
                  {systemStatus?.status || "Initializing"}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-blue-500" />
                <span className="text-muted-foreground">Agents:</span>
                <span className="font-medium">{systemStatus?.agents_count || 0}</span>
              </div>

              <div className="flex items-center gap-2">
                <MemoryStick className="w-4 h-4 text-purple-500" />
                <span className="text-muted-foreground">Tasks:</span>
                <span className="font-medium">{systemStatus?.active_tasks || 0}</span>
              </div>

              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-500" />
                <span className="text-muted-foreground">Uptime:</span>
                <span className="font-medium">{uptime}</span>
              </div>
            </div>

            <div className="text-sm text-muted-foreground">
              {time.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
