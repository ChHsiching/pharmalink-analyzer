<template>
  <div :class="['pl-input', { 'pl-input--error': error, 'pl-input--disabled': disabled }]">
    <label v-if="label" class="pl-input__label">{{ label }}</label>
    <div class="pl-input__wrapper">
      <span v-if="$slots['icon-left']" class="pl-input__icon pl-input__icon--left">
        <slot name="icon-left" />
      </span>
      <input
        :type="type"
        :value="modelValue"
        :placeholder="placeholder"
        :disabled="disabled"
        class="pl-input__field"
        @input="$emit('update:modelValue', ($event.target as HTMLInputElement).value)"
      />
      <span v-if="$slots['icon-right']" class="pl-input__icon pl-input__icon--right">
        <slot name="icon-right" />
      </span>
    </div>
    <span v-if="error" class="pl-input__error">{{ error }}</span>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  modelValue?: string | number;
  placeholder?: string;
  label?: string;
  type?: "text" | "number";
  disabled?: boolean;
  error?: string;
}>();

defineEmits<{ "update:modelValue": [value: string] }>();
</script>

<style scoped>
.pl-input {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}
.pl-input__label {
  font-size: var(--text-sm);
  font-weight: var(--text-sm-weight);
  color: var(--color-charcoal);
}
.pl-input__wrapper {
  position: relative;
  display: flex;
  align-items: center;
}
.pl-input__icon {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-stone);
  pointer-events: none;
}
.pl-input__icon--left {
  left: var(--space-3);
}
.pl-input__icon--right {
  right: var(--space-3);
}
.pl-input__field {
  width: 100%;
  padding: var(--space-2) var(--space-3);
  font-family: var(--font-family);
  font-size: var(--text-body);
  color: var(--color-ink);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  border-radius: var(--radius-md);
  outline: none;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.pl-input__field:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(61, 139, 122, 0.12);
}
.pl-input--error .pl-input__field {
  border-color: var(--color-error);
}
.pl-input--error .pl-input__field:focus {
  box-shadow: 0 0 0 3px rgba(224, 49, 49, 0.12);
}
.pl-input--disabled .pl-input__field {
  background: var(--color-surface);
  cursor: not-allowed;
  opacity: 0.6;
}
.pl-input__error {
  font-size: var(--text-xs);
  color: var(--color-error);
}
</style>
