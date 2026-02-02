<template>
  <Teleport to="body">
    <Transition name="pl-toast">
      <div v-if="message" :class="['pl-toast', `pl-toast--${variant}`]">
        <PlIcon :name="statusIcon" size="sm" class="pl-toast__icon" />
        <span class="pl-toast__message">{{ message }}</span>
        <button class="pl-toast__close" @click="dismiss">
          <PlIcon name="x" size="sm" />
        </button>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, computed } from "vue";
import PlIcon from "./PlIcon.vue";

const props = withDefaults(defineProps<{
  message?: string;
  variant?: "success" | "warning" | "error";
  duration?: number;
}>(), {
  variant: "success",
  duration: 4000,
});

const emit = defineEmits<{ "update:message": [value: ""] }>();

const statusIcon = computed(() => {
  switch (props.variant) {
    case "success": return "check";
    case "warning": return "warning";
    case "error": return "error";
    default: return "info";
  }
});

let timer: ReturnType<typeof setTimeout> | null = null;

function startTimer() {
  clearTimer();
  if (props.message && props.duration > 0) {
    timer = setTimeout(dismiss, props.duration);
  }
}

function clearTimer() {
  if (timer !== null) {
    clearTimeout(timer);
    timer = null;
  }
}

function dismiss() {
  clearTimer();
  emit("update:message", "");
}

onMounted(startTimer);
onUnmounted(clearTimer);
</script>

<style scoped>
.pl-toast {
  position: fixed;
  bottom: var(--space-6);
  right: var(--space-6);
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-md);
  font-size: var(--text-body);
  font-weight: var(--text-body-weight);
  box-shadow: var(--shadow-card);
  z-index: 9999;
}
.pl-toast--success { background: var(--color-tint-mint); color: var(--color-success); }
.pl-toast--warning { background: var(--color-tint-peach); color: var(--color-warning); }
.pl-toast--error { background: var(--color-tint-rose); color: var(--color-error); }
.pl-toast__icon {
  flex-shrink: 0;
}
.pl-toast__message {
  flex: 1;
}
.pl-toast__close {
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  opacity: 0.6;
  padding: 0;
}
.pl-toast__close:hover {
  opacity: 1;
}

/* Transition */
.pl-toast-enter-active,
.pl-toast-leave-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}
.pl-toast-enter-from,
.pl-toast-leave-to {
  opacity: 0;
  transform: translateY(8px);
}
</style>
