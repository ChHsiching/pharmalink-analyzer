import { createRouter, createWebHistory } from "vue-router";
import HomeView from "@/views/HomeView.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "home",
      component: HomeView,
    },
    {
      path: "/data-import",
      name: "data-import",
      component: () => import("@/views/DataImportView.vue"),
    },
    {
      path: "/training",
      name: "training",
      component: () => import("@/views/TrainingConfigView.vue"),
    },
    {
      path: "/analysis",
      name: "analysis",
      component: () => import("@/views/AttentionAnalysisView.vue"),
    },
    {
      path: "/expression",
      name: "expression",
      component: () => import("@/views/ExpressionDerivationView.vue"),
    },
  ],
});

export default router;
