import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";
import "katex/dist/katex.min.css";
import "./styles/design-tokens.css";

const app = createApp(App);
app.use(router);
app.mount("#app");
