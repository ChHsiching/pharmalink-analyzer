<template>
  <div class="pl-input">
    <label v-if="label" class="pl-input__label">{{ label }}</label>
    <input
      :type="type"
      :value="modelValue"
      :placeholder="placeholder"
      class="pl-input__field"
      @input="$emit('update:modelValue', ($event.target as HTMLInputElement).value)"
    />
  </div>
</template>

<script setup lang="ts">
defineProps<{
  modelValue?: string | number;
  placeholder?: string;
  label?: string;
  type?: "text" | "number";
}>();

defineEmits<{ "update:modelValue": [value: string] }>();
</script>

<style scoped>
.pl-input {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.pl-input__label {
  font-size: 13px;
  font-weight: var(--font-weight-medium);
  color: var(--color-charcoal);
}
.pl-input__field {
  padding: 8px 12px;
  font-family: var(--font-family);
  font-size: var(--font-size-body);
  color: var(--color-ink);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  border-radius: var(--radius-md);
  outline: none;
  transition: border-color 0.15s;
}
.pl-input__field:focus {
  border-color: var(--color-primary);
}
</style>
