import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import PlSelect from "@/components/PlSelect.vue";

const options = [
  { value: "a", label: "Option A" },
  { value: "b", label: "Option B" },
  { value: "c", label: "Option C" },
];

function mountSelect(overrides: Record<string, any> = {}) {
  return mount(PlSelect, {
    props: {
      options,
      modelValue: undefined,
      ...overrides,
    },
    attachTo: document.body,
  });
}

describe("PlSelect", () => {
  describe("trigger behavior", () => {
    it("opens dropdown on trigger click", async () => {
      const wrapper = mountSelect();
      await wrapper.find(".pl-select__trigger").trigger("click");
      await nextTick();
      expect(wrapper.find(".pl-select__panel").exists()).toBe(true);
    });

    it("does not open when disabled", async () => {
      const wrapper = mountSelect({ disabled: true });
      await wrapper.find(".pl-select__trigger").trigger("click");
      await nextTick();
      expect(wrapper.find(".pl-select__panel").exists()).toBe(false);
    });

    it("shows placeholder when no value selected", () => {
      const wrapper = mountSelect({ placeholder: "Pick one" });
      expect(wrapper.find(".pl-select__trigger-text").text()).toBe("Pick one");
    });

    it("shows selected option label", () => {
      const wrapper = mountSelect({ modelValue: "b" });
      expect(wrapper.find(".pl-select__trigger-text").text()).toBe("Option B");
    });
  });

  describe("option selection", () => {
    it("emits update:modelValue when option clicked", async () => {
      const wrapper = mountSelect();
      await wrapper.find(".pl-select__trigger").trigger("click");
      await nextTick();
      await wrapper.findAll(".pl-select__option")[1].trigger("click");
      expect(wrapper.emitted("update:modelValue")).toEqual([["b"]]);
    });

    it("closes dropdown after selecting an option", async () => {
      const wrapper = mountSelect();
      await wrapper.find(".pl-select__trigger").trigger("click");
      await nextTick();
      expect(wrapper.find(".pl-select__panel").exists()).toBe(true);
      await wrapper.findAll(".pl-select__option")[0].trigger("click");
      await nextTick();
      expect(wrapper.find(".pl-select__panel").exists()).toBe(false);
    });

    it("applies selected class to the current option", async () => {
      const wrapper = mountSelect({ modelValue: "a" });
      await wrapper.find(".pl-select__trigger").trigger("click");
      await nextTick();
      const opts = wrapper.findAll(".pl-select__option");
      expect(opts[0].classes()).toContain("pl-select__option--selected");
      expect(opts[1].classes()).not.toContain("pl-select__option--selected");
    });
  });

  describe("closing behavior", () => {
    it("closes on Escape key", async () => {
      const wrapper = mountSelect();
      await wrapper.find(".pl-select__trigger").trigger("click");
      await nextTick();
      expect(wrapper.find(".pl-select__panel").exists()).toBe(true);
      await wrapper.find(".pl-select__trigger").trigger("keydown.escape");
      await nextTick();
      expect(wrapper.find(".pl-select__panel").exists()).toBe(false);
    });

    it("closes on click outside", async () => {
      const wrapper = mountSelect();
      await wrapper.find(".pl-select__trigger").trigger("click");
      await nextTick();
      expect(wrapper.find(".pl-select__panel").exists()).toBe(true);
      document.dispatchEvent(new MouseEvent("click"));
      await nextTick();
      expect(wrapper.find(".pl-select__panel").exists()).toBe(false);
    });
  });
});
