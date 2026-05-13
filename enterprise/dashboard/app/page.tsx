"use client";

import { useState, useEffect } from "react";
import { AgentCard } from "@/components/AgentCard";
import { MetricsPanel } from "@/components/MetricsPanel";
import { EventStream } from "@/components/EventStream";
import { TaskGraph } from "@/components/TaskGraph";
import { AgentChat } from "@/components/AgentChat";
import { SystemHealth } from "@/components/SystemHealth";
import { Header } from "@/components/Header";

export default function Dashboard() {
  const [agents, setAgents] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any>({});
  const [events, setEvents] = useState<any[]>([]);
  const [systemStatus, setSystemStatus] = useState<any>({});
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [agentsRes, statusRes, eventsRes] = await Promise.all([
          fetch("/api/agents"),
          fetch("/api/status"),
          fetch("/api/events?limit=50"),
        ]);

        if (agentsRes.ok) {
          const data = await agentsRes.json();
          setAgents(data.agents || []);
        }

        if (statusRes.ok) {
          const data = await statusRes.json();
          setSystemStatus(data);
        }

        if (eventsRes.ok) {
          const data = await eventsRes.json();
          setEvents(data.events || []);
        }
      } catch (error) {
        console.error("Failed to fetch data:", error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header systemStatus={systemStatus} />

      <main className="container mx-auto p-4 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <AgentCard
              agents={agents}
              selectedAgent={selectedAgent}
              onSelectAgent={setSelectedAgent}
            />
          </div>
          <div className="space-y-6">
            <SystemHealth status={systemStatus} />
            <MetricsPanel metrics={metrics} />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TaskGraph />
          <EventStream events={events} />
        </div>

        {selectedAgent && (
          <AgentChat agentId={selectedAgent} />
        )}
      </main>
    </div>
  );
}
