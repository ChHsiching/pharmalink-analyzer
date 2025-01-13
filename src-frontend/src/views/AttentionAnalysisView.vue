<!-- src/views/AttentionAnalysisView.vue -->
<template>
  <div class="attention-analysis">
    <h2>注意力分析</h2>
    <div class="toolbar">
      <select v-model="selectedCheckpoint" class="checkpoint-select">
        <option value="">选择检查点</option>
        <option
          v-for="cp in checkpoints"
          :key="cp.id"
          :value="cp.id"
        >
          {{ cp.id }} (loss: {{ cp.final_loss.toFixed(4) }})
        </option>
      </select>
      <button
        @click="analyze"
        :disabled="!selectedCheckpoint || loading"
        class="btn-analyze"
      >
        分析
      </button>
      <label v-if="network" class="threshold-control">
        阈值: {{ threshold }}
        <input
          type="range"
          v-model.number="threshold"
          min="0"
          max="100"
          step="1"
          @change="updateNetwork"
        />
      </label>
    </div>

    <div v-if="loading" class="loading">加载中...</div>
    <div v-else-if="error" class="error">{{ error }}</div>

    <div v-if="heatmap" class="panels">
      <div class="panel">
        <h3>注意力热力图</h3>
        <v-chart :option="heatmapOption" autoresize style="height: 500px" />
      </div>
      <div class="panel">
        <h3>关联网络图</h3>
        <div ref="networkRef" class="network-container"></div>
        <div class="legend">
          <span class="legend-item synergistic">协同</span>
          <span class="legend-item antagonistic">拮抗</span>
        </div>
      </div>
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
        color: ["#313695", "#4575b4", "#74add1", "#abd9e9", "#fee090", "#fdae61", "#f46d43", "#d73027"],
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
      d.classification === "synergistic" ? "#4caf50" : "#f44336",
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
    .attr("fill", "#2196f3")
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
  color: #2e7d32;
  text-align: center;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  justify-content: center;
}

.checkpoint-select {
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  min-width: 250px;
}

.btn-analyze {
  padding: 6px 20px;
  background: #4caf50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.btn-analyze:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.threshold-control {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.panels {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.panel {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 16px;
}

.panel h3 {
  margin: 0 0 12px 0;
  color: #333;
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
  background: #4caf50;
}

.legend-item.antagonistic::before {
  content: "";
  display: inline-block;
  width: 12px;
  height: 3px;
  background: #f44336;
}

.loading,
.error {
  text-align: center;
  padding: 40px;
  color: #666;
}

.error {
  color: #f44336;
}
</style>
