import { ref, onUnmounted } from "vue";

const WEEKDAYS = ["日", "一", "二", "三", "四", "五", "六"] as const;

function formatTime(date: Date): string {
  const y = date.getFullYear();
  const mo = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const w = WEEKDAYS[date.getDay()];
  const h = String(date.getHours()).padStart(2, "0");
  const m = String(date.getMinutes()).padStart(2, "0");
  const s = String(date.getSeconds()).padStart(2, "0");
  return `${y}/${mo}/${d} 星期${w} ${h}:${m}:${s}`;
}

export function useLiveClock() {
  const timeString = ref(formatTime(new Date()));
  let timer: ReturnType<typeof setInterval> | null = null;

  function tick(): void {
    timeString.value = formatTime(new Date());
  }

  timer = setInterval(tick, 1000);

  function stop(): void {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  onUnmounted(stop);

  return { timeString, stop };
}
