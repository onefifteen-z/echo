import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import type { EchoPromptAPI, MeetingPromptData } from "@shared/types";
import iconUrl from "../../assets/icons/png/128x128.png";
import "./styles.css";

declare global {
  interface Window {
    echoPrompt: EchoPromptAPI;
  }
}

function Prompt() {
  const [data, setData] = useState<MeetingPromptData | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(
    () =>
      window.echoPrompt.onShow((d) => {
        setMenuOpen(false);
        setData(d);
      }),
    [],
  );
  if (!data) return null;
  const act = window.echoPrompt.act;
  const toggleMenu = (open: boolean) => {
    setMenuOpen(open);
    act(open ? "menu-open" : "menu-close");
  };

  return (
    <div
      className="group relative h-[112px] p-2"
      onMouseEnter={() => act("hover-start")}
      onMouseLeave={() => !menuOpen && act("hover-end")}
    >
      <div className="flex h-full flex-col overflow-hidden rounded-2xl border border-white/15 bg-[#1b1d22]/95 shadow-2xl">
        <div className="flex flex-1 items-center gap-3 px-4">
          <img src={iconUrl} alt="" className="h-10 w-10 shrink-0 rounded-[10px]" />
          <div className="min-w-0 flex-1">
            <div className="truncate text-[15px] font-semibold text-zinc-100">{data.appName} meeting detected</div>
            <div className="truncate text-[13px] text-zinc-400" title={data.title}>
              {data.title}
            </div>
          </div>
          <div className="flex shrink-0 overflow-hidden rounded-full bg-accent text-black">
            <button className="px-4 py-2 text-sm font-semibold hover:brightness-110" onClick={() => act("start")}>
              Start transcribing
            </button>
            <button
              aria-label="More options"
              className="border-l border-black/20 px-2.5 hover:brightness-110"
              aria-expanded={menuOpen}
              onClick={() => toggleMenu(!menuOpen)}
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M2.5 4.5 6 8l3.5-3.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
        </div>
        <div className="truncate border-t border-white/10 bg-black/20 px-4 py-1.5 text-[11px] text-zinc-500">
          Transcribing with Echo — make sure everyone in the meeting agrees to being recorded.
        </div>
      </div>
      <button
        aria-label="Dismiss"
        onClick={() => act("dismiss")}
        className="absolute top-0 left-0 flex h-6 w-6 items-center justify-center rounded-full border border-white/20 bg-[#2a2d33] text-zinc-300 opacity-0 transition group-hover:opacity-100 hover:text-white"
      >
        <svg width="10" height="10" viewBox="0 0 10 10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
          <path d="M2 2l6 6M8 2 2 8" />
        </svg>
      </button>
      {menuOpen && (
        <div
          role="menu"
          className="absolute top-[100px] right-2 w-60 overflow-hidden rounded-xl border border-white/15 bg-[#1b1d22] py-1 text-[13px] text-zinc-200 shadow-2xl"
        >
          <MenuItem onClick={() => act("setup")}>Set up first…</MenuItem>
          <div className="my-1 border-t border-white/10" />
          <MenuItem onClick={() => act("mute")}>Don't ask for {data.appName}</MenuItem>
          <MenuItem onClick={() => act("disable")}>Turn off meeting detection</MenuItem>
        </div>
      )}
    </div>
  );
}

function MenuItem({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button role="menuitem" className="block w-full px-3 py-1.5 text-left hover:bg-white/10" onClick={onClick}>
      {children}
    </button>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Prompt />
  </React.StrictMode>,
);
