"use client";

import { useState } from "react";
import { Activity, Brain, Crosshair, Zap } from "lucide-react";

interface Event {
  id: string;
  type: string;
  timestamp: string;
  source: string;
  data: any;
}

interface EventStreamProps {
  events: Event[];
}

const eventIcons: Record<string, any> = {
  agent_message: Brain,
  task_created: Zap,
  task_completed: CheckCircle,
  decision_proposed: Crosshair,
  vision_update: Activity,
  learning_update: Brain,
  error: AlertCircle,
};

const eventColors: Record<string, string> = {
  agent_message: "text-blue-500",
  task_created: "text-purple-500",
  task_completed: "text-green-500",
  decision_proposed: "text-yellow-500",
  vision_update: "text-cyan-500",
  learning_update: "text-violet-500",
  error: "text-red-500",
};

export function EventStream({ events }: EventStreamProps) {
  const [filter, setFilter] = useState<string>("all");
  const [autoScroll, setAutoScroll] = useState(true);

  const filteredEvents = filter === "all"
    ? events
    : events.filter((e) => e.type === filter);

  const sampleEvents: Event[] = [
    {
      id: "1",
      type: "agent_message",
      timestamp: new Date().toISOString(),
      source: "strategy-agent",
      data: { action: "plan_created" },
    },
    {
      id: "2",
      type: "task_created",
      timestamp: new Date().toISOString(),
      source: "supervisor",
      data: { task_id: "task_123" },
    },
    {
      id: "3",
      type: "decision_proposed",
      timestamp: new Date().toISOString(),
      source: "combat-agent",
      data: { decision: "engage_target" },
    },
  ];

  const displayEvents = events.length > 0 ? filteredEvents : sampleEvents;

  return (
    <div className="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-500" />
          Event Stream
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`px-3 py-1 text-xs rounded-full transition-colors ${
              autoScroll
                ? "bg-green-500/20 text-green-500"
                : "bg-secondary text-muted-foreground"
            }`}
          >
            Auto-scroll
          </button>
        </div>
      </div>

      <div className="space-y-2 max-h-80 overflow-y-auto">
        {displayEvents.map((event) => {
          const Icon = eventIcons[event.type] || Activity;
          const colorClass = eventColors[event.type] || "text-muted-foreground";

          return (
            <div
              key={event.id}
              className="flex items-start gap-3 p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
            >
              <Icon className={`w-4 h-4 mt-0.5 ${colorClass}`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium capitalize">
                    {event.type.replace(/_/g, " ")}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {event.source}
                </p>
                {event.data && (
                  <pre className="text-xs text-muted-foreground/70 mt-2 p-2 rounded bg-background/50 overflow-x-auto">
                    {JSON.stringify(event.data, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {displayEvents.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No events yet</p>
        </div>
      )}
    </div>
  );
}

function CheckCircle(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function AlertCircle(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}
