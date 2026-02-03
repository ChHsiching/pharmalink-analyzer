<!-- src/views/AttentionAnalysisView.vue -->
<template>
  <div class="attention-analysis">
    <PlPageHeader
      :step="3"
      title="注意力分析"
      subtitle="可视化分析 Transformer 注意力权重与成分关联"
    />

    <!-- Toolbar -->
    <div class="attention-analysis__toolbar">
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

    <!-- Status -->
    <PlSpinner v-if="loading" size="md" />
    <PlToast v-else-if="error" :message="error" variant="error" />

    <PlEmptyState
      v-else-if="!heatmap && !network"
      title="尚无分析数据"
      description="选择一个检查点并点击分析以查看注意力热力图和关联网络图"
    />

    <!-- Visualization panels: 2-col equal grid -->
    <div v-if="heatmap" class="attention-analysis__viz-grid">
      <PlCard variant="base" padding="md">
        <h3 class="card-title">注意力热力图</h3>
        <v-chart :option="heatmapOption" theme="pharmalink" autoresize class="heatmap-chart" />
      </PlCard>
      <PlCard variant="base" padding="md">
        <h3 class="card-title">关联网络图</h3>
        <div ref="networkRef" class="network-container"></div>
        <div class="network-legend">
          <span class="network-legend__item network-legend__item--synergistic">协同</span>
          <span class="network-legend__item network-legend__item--antagonistic">拮抗</span>
        </div>
      </PlCard>
    </div>

    <!-- Summary bar: Top-5 attention weights -->
    <PlCard v-if="topAttentionWeights.length" variant="base" padding="md">
      <h3 class="card-title">Top-5 注意力权重</h3>
      <div class="attention-analysis__summary">
        <div
          v-for="(item, idx) in topAttentionWeights"
          :key="idx"
          class="attention-analysis__summary-row"
        >
          <span class="attention-analysis__summary-label">
            {{ item.source }} &rarr; {{ item.target }}
          </span>
          <PlProgressBar
            :value="item.weight"
            :max="maxAttentionValue"
            :label="item.weight.toFixed(4)"
          />
        </div>
      </div>
    </PlCard>

    <!-- Footer navigation -->
    <div class="attention-analysis__footer">
      <PlButton variant="secondary" @click="goPrev">
        <PlIcon name="arrow-left" size="sm" />
        上一步：模型训练
      </PlButton>
      <PlButton variant="dark" :disabled="!heatmap" @click="goNext">
        下一步：表达式推导
        <PlIcon name="arrow-right" size="sm" />
      </PlButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onUnmounted, nextTick } from "vue";
import { useRouter } from "vue-router";
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
import PlPageHeader from "@/components/PlPageHeader.vue";
import PlIcon from "@/components/PlIcon.vue";
import PlProgressBar from "@/components/PlProgressBar.vue";

use([HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

const router = useRouter();

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

// Top-5 attention weights extracted from heatmap matrix
interface AttentionEntry {
  source: string;
  target: string;
  weight: number;
}

const topAttentionWeights = computed<AttentionEntry[]>(() => {
  if (!heatmap.value) return [];
  const names = heatmap.value.feature_names;
  const values = heatmap.value.values;
  const entries: AttentionEntry[] = [];
  for (let i = 0; i < names.length; i++) {
    for (let j = 0; j < names.length; j++) {
      if (i !== j) {
        entries.push({ source: names[i], target: names[j], weight: values[i][j] });
      }
    }
  }
  return entries
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 5);
});

const maxAttentionValue = computed(() => {
  if (topAttentionWeights.value.length === 0) return 1;
  return topAttentionWeights.value[0].weight;
});

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
        color: ["#1aae39", "#8bc34a", "#f5d75e", "#ff9800", "#e03131"],
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
    .attr("fill", getComputedStyle(document.documentElement).getPropertyValue("--color-primary").trim() || CHART_COLORS.primary)
    .attr("stroke", getComputedStyle(document.documentElement).getPropertyValue("--color-hairline").trim() || "#e5e3df")
    .attr("stroke-width", 2);

  node
    .append("text")
    .text((d) => d.name)
    .attr("font-size", getComputedStyle(document.documentElement).getPropertyValue("--text-sm").trim() || "0.9286rem")
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

function goPrev() {
  router.push({ name: "training" });
}

function goNext() {
  router.push({ name: "expression" });
}
</script>

<style scoped>
.attention-analysis {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: var(--space-10);
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
  font-family: var(--font-family);
}

/* ── Toolbar ── */
.attention-analysis__toolbar {
  display: flex;
  align-items: flex-end;
  gap: var(--space-3);
  flex-wrap: wrap;
}

/* ── Visualization grid (1:1 split) ── */
.attention-analysis__viz-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-6);
}

@media (max-width: 1024px) {
  .attention-analysis__viz-grid {
    grid-template-columns: 1fr;
  }
}

/* ── Card title ── */
.card-title {
  font-size: var(--text-h3);
  font-weight: var(--text-h3-weight);
  color: var(--color-charcoal);
  margin: 0 0 var(--space-4);
}

/* ── Heatmap chart ── */
.heatmap-chart {
  height: 500px;
  width: 100%;
}

/* ── Network graph ── */
.network-container {
  width: 100%;
  height: 500px;
}

.network-legend {
  display: flex;
  gap: var(--space-4);
  margin-top: var(--space-2);
  justify-content: center;
}

.network-legend__item {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: var(--text-sm);
  color: var(--color-slate);
}

.network-legend__item--synergistic::before {
  content: "";
  display: inline-block;
  width: 12px;
  height: 3px;
  background: var(--color-success);
  border-radius: var(--radius-full);
}

.network-legend__item--antagonistic::before {
  content: "";
  display: inline-block;
  width: 12px;
  height: 3px;
  background: var(--color-error);
  border-radius: var(--radius-full);
}

/* ── Summary bar: Top-5 attention weights ── */
.attention-analysis__summary {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.attention-analysis__summary-row {
  display: grid;
  grid-template-columns: 220px 1fr;
  align-items: center;
  gap: var(--space-4);
}

.attention-analysis__summary-label {
  font-size: var(--text-sm);
  color: var(--color-charcoal);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

/* ── Footer navigation ── */
.attention-analysis__footer {
  display: flex;
  justify-content: space-between;
  padding-top: var(--space-4);
}
</style>
