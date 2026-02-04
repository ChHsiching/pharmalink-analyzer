<template>
  <div class="pl-select" ref="container" :style="{ minWidth: `${minWidth}px` }">
    <label v-if="label" class="pl-select__label">{{ label }}</label>
    <div
      class="pl-select__trigger"
      :class="{ 'pl-select__trigger--disabled': disabled, 'pl-select__trigger--open': open }"
      tabindex="0"
      @click="toggle"
      @keydown.escape="close"
    >
      <span class="pl-select__trigger-text" :class="{ 'pl-select__trigger-text--placeholder': !selectedLabel }">
        {{ selectedLabel || placeholder }}
      </span>
      <svg class="pl-select__chevron" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">
        <path fill="currentColor" d="M2 4l4 4 4-4" />
      </svg>
    </div>
    <div v-if="open" class="pl-select__panel">
      <div
        v-for="opt in options"
        :key="opt.value"
        class="pl-select__option"
        :class="{ 'pl-select__option--selected': opt.value === modelValue }"
        @click="select(opt.value)"
      >
        {{ opt.label }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from "vue";

const props = withDefaults(
  defineProps<{
    modelValue?: string;
    options: { value: string; label: string }[];
    label?: string;
    placeholder?: string;
    disabled?: boolean;
  }>(),
  { placeholder: "请选择", disabled: false },
);

const emit = defineEmits<{ "update:modelValue": [value: string] }>();

const open = ref(false);
const container = ref<HTMLElement>();

const selectedLabel = computed(
  () => props.options.find((o) => o.value === props.modelValue)?.label,
);

const minWidth = computed(() => {
  if (typeof document === "undefined") return 160;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return 160;
  ctx.font = '14px "Notion Sans", Inter, system-ui, sans-serif';
  const labels = props.options.map((o) => o.label);
  if (props.placeholder) labels.push(props.placeholder);
  const longest = labels.reduce((a, b) => (a.length > b.length ? a : b), "");
  const textWidth = ctx.measureText(longest).width;
  return Math.max(Math.ceil(textWidth) + 44, 160);
});

function toggle() {
  if (props.disabled) return;
  open.value = !open.value;
}

function close() {
  open.value = false;
}

function select(value: string) {
  emit("update:modelValue", value);
  close();
}

function onClickOutside(e: MouseEvent) {
  if (container.value && !container.value.contains(e.target as Node)) {
    close();
  }
}

onMounted(() => document.addEventListener("click", onClickOutside));
onUnmounted(() => document.removeEventListener("click", onClickOutside));
</script>

<style scoped>
.pl-select {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  position: relative;
}

.pl-select__label {
  font-size: var(--text-sm);
  font-weight: var(--text-sm-weight);
  color: var(--color-charcoal);
}

.pl-select__trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-2) var(--space-3);
  padding-right: 32px;
  font-family: var(--font-family);
  font-size: var(--text-body);
  color: var(--color-ink);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
  position: relative;
  user-select: none;
}

.pl-select__trigger:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(61, 139, 122, 0.12);
  outline: none;
}

.pl-select__trigger--disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.pl-select__trigger--open {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(61, 139, 122, 0.12);
}

.pl-select__trigger-text {
  color: var(--color-ink);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pl-select__trigger-text--placeholder {
  color: var(--color-steel);
}

.pl-select__chevron {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--color-steel);
  pointer-events: none;
}

.pl-select__trigger--open .pl-select__chevron {
  transform: translateY(-50%) rotate(180deg);
}

.pl-select__panel {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 50;
  width: 100%;
  max-height: 240px;
  overflow-y: auto;
  margin-top: var(--space-1);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  border-radius: var(--radius-md);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.pl-select__option {
  padding: 8px 12px;
  font-size: var(--text-body);
  color: var(--color-charcoal);
  cursor: pointer;
  transition: background-color 0.1s;
}

.pl-select__option:hover {
  background: var(--color-surface-soft);
}

.pl-select__option--selected {
  background: var(--color-primary);
  color: var(--color-on-primary);
}

.pl-select__option--selected:hover {
  background: var(--color-primary);
}
</style>
