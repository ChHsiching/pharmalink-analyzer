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
  gap: 4px;
}
.pl-select__label {
  font-size: 13px;
  font-weight: var(--font-weight-medium);
  color: var(--color-charcoal);
}
.pl-select__field {
  padding: 8px 12px;
  font-family: var(--font-family);
  font-size: var(--font-size-body);
  color: var(--color-ink);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  border-radius: var(--radius-md);
  outline: none;
  cursor: pointer;
  transition: border-color 0.15s;
}
.pl-select__field:focus {
  border-color: var(--color-primary);
}
</style>
