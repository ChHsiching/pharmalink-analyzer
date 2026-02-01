<!-- src/views/AttentionAnalysisView.vue -->
<template>
  <div class="attention-analysis">
    <h2>注意力分析</h2>
    <div class="toolbar">
      <PlSelect
        :model-value="selectedCheckpoint"
        :options="checkpointOptions"
        label="检查点"
        @update:model-value="selectedCheckpoint = $event"
      />
      <PlButton
        variant="primary"
        :disabled="!selectedCheckpoint || loading"
        @click="analyze"
      >
        分析
      </PlButton>
      <PlInput
        v-if="network"
        :model-value="String(threshold)"
        type="number"
        label="阈值"
        @update:model-value="handleThresholdChange"
      />
    </div>

    <PlSpinner v-if="loading" size="md" />
    <PlToast v-else-if="error" :message="error" variant="error" />

    <PlEmptyState
      v-else-if="!heatmap && !network"
      title="尚无分析数据"
      description="选择一个检查点并点击分析以查看注意力热力图和关联网络图"
    />

    <div v-if="heatmap" class="panels">
      <PlCard variant="base" padding="md">
        <h3>注意力热力图</h3>
        <v-chart :option="heatmapOption" autoresize style="height: 500px" />
      </PlCard>
      <PlCard variant="base" padding="md">
        <h3>关联网络图</h3>
        <div ref="networkRef" class="network-container"></div>
        <div class="legend">
          <span class="legend-item synergistic">协同</span>
          <span class="legend-item antagonistic">拮抗</span>
        </div>
      </PlCard>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onUnmounted, nextTick } from "vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { HeatmapChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import * as d3 from "d3";
import type { NetworkGraphResponse, NetworkNode } from "@/types/analysis";
import { useAnalysis } from "@/composables/useAnalysis";
import { useTraining } from "@/composables/useTraining";
import { CHART_COLORS } from "@/utils/chart-palette";

import PlButton from "@/components/PlButton.vue";
import PlSelect from "@/components/PlSelect.vue";
import PlInput from "@/components/PlInput.vue";
import PlCard from "@/components/PlCard.vue";
import PlSpinner from "@/components/PlSpinner.vue";
import PlToast from "@/components/PlToast.vue";
import PlEmptyState from "@/components/PlEmptyState.vue";

use([HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

interface SimNode extends NetworkNode {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

const { heatmap, network, loading, error, fetchHeatmap, fetchNetwork } =
  useAnalysis();
const { checkpoints, fetchCheckpoints } = useTraining();

const selectedCheckpoint = ref("");
const threshold = ref(50);
const networkRef = ref<HTMLElement | null>(null);
let simulation: d3.Simulation<SimNode, d3.SimulationLinkDatum<SimNode>> | null =
  null;

fetchCheckpoints();

const checkpointOptions = computed(() =>
  checkpoints.value.map((cp) => ({
    value: cp.id,
    label: `${cp.id} (loss: ${cp.final_loss.toFixed(4)})`,
  })),
);

const heatmapOption = computed(() => {
  if (!heatmap.value) return {};
  const names = heatmap.value.feature_names;
  const n = names.length;
  const data: [number, number, number][] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      data.push([i, j, heatmap.value.values[i][j]]);
    }
  }
  return {
    tooltip: {
      position: "top",
      formatter: (params: any) => {
        const [i, j, val] = params.data;
        return `${names[i]} → ${names[j]}: ${val.toFixed(4)}`;
      },
    },
    grid: { top: "10%", bottom: "20%", left: "15%", right: "10%" },
    xAxis: {
      type: "category",
      data: names,
      splitArea: { show: true },
      axisLabel: { rotate: 45, fontSize: 10 },
    },
    yAxis: {
      type: "category",
      data: names,
      splitArea: { show: true },
      axisLabel: { fontSize: 10 },
    },
    visualMap: {
      min: heatmap.value.min_value,
      max: heatmap.value.max_value,
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: "0%",
      inRange: {
        color: ["#dcecfa", "#5a8fa8", "#3d8b7a", "#dd5b00", "#7b3ff2", "#5b6abf", "#e03131", "#1a1a1a"],
      },
    },
    series: [
      {
        type: "heatmap",
        data,
        label: { show: n <= 15, fontSize: 8 },
        emphasis: {
          itemStyle: { shadowBlur: 10, shadowColor: "rgba(0, 0, 0, 0.5)" },
        },
      },
    ],
  };
});

function renderNetworkGraph(
  container: HTMLElement,
  data: NetworkGraphResponse,
) {
  d3.select(container).select("svg").remove();

  const width = container.clientWidth || 500;
  const height = 500;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const nodes: SimNode[] = data.nodes.map((n) => ({
    ...n,
    x: width / 2 + (Math.random() - 0.5) * 50,
    y: height / 2 + (Math.random() - 0.5) * 50,
  }));
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const links = data.edges.map((e) => ({
    source: nodeMap.get(e.source)!,
    target: nodeMap.get(e.target)!,
    weight: e.weight,
    classification: e.classification,
  }));

  simulation = d3
    .forceSimulation<SimNode>(nodes)
    .force(
      "link",
      d3
        .forceLink<SimNode, typeof links[0]>(links)
        .id((d) => d.id)
        .distance(80),
    )
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(20));

  const link = svg
    .append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", (d) =>
      d.classification === "synergistic" ? CHART_COLORS.success : CHART_COLORS.error,
    )
    .attr("stroke-width", (d) => Math.max(1, d.weight * 10))
    .attr("stroke-opacity", 0.6);

  const node = svg
    .append("g")
    .selectAll<SVGGElement, SimNode>("g")
    .data(nodes)
    .join("g")
    .call(
      d3
        .drag<SVGGElement, SimNode>()
        .on("start", (event, d) => {
          if (!event.active) simulation!.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation!.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }),
    );

  node
    .append("circle")
    .attr("r", 10)
    .attr("fill", CHART_COLORS.primary)
    .attr("stroke", "#fff")
    .attr("stroke-width", 2);

  node
    .append("text")
    .text((d) => d.name)
    .attr("font-size", 10)
    .attr("dx", 14)
    .attr("dy", 4);

  simulation.on("tick", () => {
    link
      .attr("x1", (d) => (d.source as SimNode).x!)
      .attr("y1", (d) => (d.source as SimNode).y!)
      .attr("x2", (d) => (d.target as SimNode).x!)
      .attr("y2", (d) => (d.target as SimNode).y!);
    node.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });
}

function handleThresholdChange(val: string) {
  threshold.value = Number(val);
  updateNetwork();
}

async function analyze() {
  if (!selectedCheckpoint.value) return;
  await Promise.all([
    fetchHeatmap(selectedCheckpoint.value),
    fetchNetwork(selectedCheckpoint.value),
  ]);
  await nextTick();
  if (network.value && networkRef.value) {
    renderNetworkGraph(networkRef.value, network.value);
  }
}

async function updateNetwork() {
  if (!selectedCheckpoint.value) return;
  await fetchNetwork(selectedCheckpoint.value, threshold.value);
  await nextTick();
  if (network.value && networkRef.value) {
    renderNetworkGraph(networkRef.value, network.value);
  }
}

watch(network, (data) => {
  if (data && networkRef.value) {
    renderNetworkGraph(networkRef.value, data);
  }
});

onUnmounted(() => {
  simulation?.stop();
});
</script>

<style scoped>
.attention-analysis {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: system-ui, sans-serif;
}

h2 {
  color: var(--color-ink);
  text-align: center;
}

.toolbar {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  margin-bottom: 20px;
  justify-content: center;
}

.panels {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.panel h3,
.pl-card h3 {
  margin: 0 0 12px 0;
  color: var(--color-charcoal);
  font-size: 16px;
}

.network-container {
  width: 100%;
  height: 500px;
}

.legend {
  display: flex;
  gap: 16px;
  margin-top: 8px;
  justify-content: center;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
}

.legend-item.synergistic::before {
  content: "";
  display: inline-block;
  width: 12px;
  height: 3px;
  background: var(--color-success);
}

.legend-item.antagonistic::before {
  content: "";
  display: inline-block;
  width: 12px;
  height: 3px;
  background: var(--color-error);
}

@media (max-width: 1024px) {
  .panels {
    grid-template-columns: 1fr;
  }
}
</style>
