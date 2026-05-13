"use client";

import { Activity, Cpu, MemoryStick, Clock } from "lucide-react";

interface MetricsPanelProps {
  metrics: any;
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  const sampleMetrics = {
    cpu: 45,
    memory: 62,
    gpu: 78,
    latency: 23,
    throughput: 1250,
    errors: 0,
  };

  return (
    <div className="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-6">
      <h2 className="text-lg font-semibold flex items-center gap-2 mb-6">
        <Activity className="w-5 h-5 text-green-500" />
        System Metrics
      </h2>

      <div className="space-y-4">
        <MetricItem
          icon={Cpu}
          label="CPU Usage"
          value={sampleMetrics.cpu}
          color="bg-blue-500"
        />
        <MetricItem
          icon={MemoryStick}
          label="Memory"
          value={sampleMetrics.memory}
          color="bg-purple-500"
        />
        <MetricItem
          icon={Activity}
          label="GPU Load"
          value={sampleMetrics.gpu}
          color="bg-green-500"
        />
        <MetricItem
          icon={Clock}
          label="Latency"
          value={sampleMetrics.latency}
          unit="ms"
          color="bg-yellow-500"
          max={100}
        />

        <div className="pt-4 border-t border-border">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Throughput</span>
            <span className="font-medium">{sampleMetrics.throughput} req/s</span>
          </div>
          <div className="flex items-center justify-between text-sm mt-2">
            <span className="text-muted-foreground">Error Rate</span>
            <span className="font-medium text-green-500">
              {sampleMetrics.errors}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricItem({
  icon: Icon,
  label,
  value,
  unit = "%",
  color,
  max = 100,
}: {
  icon: any;
  label: string;
  value: number;
  unit?: string;
  color: string;
  max?: number;
}) {
  const percentage = Math.min((value / max) * 100, 100);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">{label}</span>
        </div>
        <span className="text-sm font-medium">
          {value}
          {unit}
        </span>
      </div>
      <div className="h-2 rounded-full bg-secondary overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
