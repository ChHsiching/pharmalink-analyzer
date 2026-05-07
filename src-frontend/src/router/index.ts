import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      redirect: "/data-import",
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
    {
      path: "/evaluation",
      name: "evaluation",
      component: () => import("@/views/EvaluationView.vue"),
    },
  ],
});

export default router;
