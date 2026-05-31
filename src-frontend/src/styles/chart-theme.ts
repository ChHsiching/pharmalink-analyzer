import { registerTheme } from "echarts/core";

const PHARMALINK_THEME = {
  color: ["#3d8b7a", "#2a9d99", "#6bb5a5", "#dd5b00", "#ff64c8", "#0075de", "#f5d75e"],
  textStyle: { fontFamily: "Inter, system-ui, sans-serif" },
  title: { textStyle: { color: "#37352f", fontSize: 14, fontWeight: 600 } },
  tooltip: {
    backgroundColor: "#1a1a1a",
    textStyle: { color: "#ffffff", fontSize: 12 },
    borderRadius: 8,
    borderWidth: 0,
    shadowBlur: 10,
    shadowColor: "rgba(0,0,0,0.15)",
  },
  grid: { containLabel: true },
  xAxis: {
    axisLine: { lineStyle: { color: "#e5e3df" } },
    splitLine: { lineStyle: { color: "#ede9e4" } },
    axisLabel: { color: "#787671" },
  },
  yAxis: {
    axisLine: { lineStyle: { color: "#e5e3df" } },
    splitLine: { lineStyle: { color: "#ede9e4" } },
    axisLabel: { color: "#787671" },
  },
  legend: { textStyle: { color: "#5d5b54" } },
};

export function registerPharmalinkTheme() {
  registerTheme("pharmalink", PHARMALINK_THEME);
}

export { PHARMALINK_THEME };
