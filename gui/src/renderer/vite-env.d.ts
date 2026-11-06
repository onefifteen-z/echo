/// <reference types="vite/client" />

import type { EchoAPI } from "@shared/types";

declare global {
  interface Window {
    echo: EchoAPI;
  }
}
