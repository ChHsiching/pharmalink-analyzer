<template>
  <div class="pl-select">
    <label v-if="label" class="pl-select__label">{{ label }}</label>
    <select
      :value="modelValue"
      class="pl-select__field"
      @change="$emit('update:modelValue', ($event.target as HTMLSelectElement).value)"
    >
      <option v-for="opt in options" :key="opt.value" :value="opt.value">
        {{ opt.label }}
      </option>
    </select>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  modelValue?: string;
  options: { value: string; label: string }[];
  label?: string;
}>();

defineEmits<{ "update:modelValue": [value: string] }>();
</script>

<style scoped>
.pl-select {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}
.pl-select__label {
  font-size: var(--text-sm);
  font-weight: var(--text-sm-weight);
  color: var(--color-charcoal);
}
.pl-select__field {
  padding: var(--space-2) var(--space-3);
  font-family: var(--font-family);
  font-size: var(--text-body);
  color: var(--color-ink);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  border-radius: var(--radius-md);
  outline: none;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23787671' d='M2 4l4 4 4-4'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  padding-right: 32px;
}
.pl-select__field:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(61, 139, 122, 0.12);
}
</style>
