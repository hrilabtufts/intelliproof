"use client";

import { useEffect, useRef, useState } from "react";
import {
  CheckIcon,
  ChevronDownIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import type { AcademicDatabase } from "../../types/autoExplore";

export type { AcademicDatabase } from "../../types/autoExplore";

const databases: { id: AcademicDatabase; label: string }[] = [
  { id: "arxiv", label: "Arxiv" },
  { id: "semantic_scholar", label: "Semantic Scholar" },
  { id: "pubmed", label: "Pubmed" },
];

type AutoExploreMenuProps = {
  selectedDatabases: AcademicDatabase[];
  onToggleDatabase: (database: AcademicDatabase) => void;
  onSearch: () => void;
  isSearching?: boolean;
  disabled?: boolean;
};

export default function AutoExploreMenu({
  selectedDatabases,
  onToggleDatabase,
  onSearch,
  isSearching = false,
  disabled = false,
}: AutoExploreMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) return;

    const handlePointerDown = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsOpen(false);
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  return (
    <div ref={menuRef} className="relative">
      <button
        type="button"
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        aria-controls={isOpen ? "auto-explore-database-menu" : undefined}
        onClick={() => setIsOpen((open) => !open)}
        className="inline-flex items-center gap-1.5 rounded-md border border-[#4b5563] bg-gradient-to-r from-[#374151] to-[#232F3E] px-2.5 py-1.5 text-sm font-medium text-[#F3F4F6] shadow-sm transition-all hover:from-[#2f3b4a] hover:to-[#1A2330] hover:shadow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#64748b]"
      >
        <MagnifyingGlassIcon className="h-4 w-4" aria-hidden="true" />
        <span>Auto-Explore</span>
        <ChevronDownIcon
          className={`h-3.5 w-3.5 transition-transform ${isOpen ? "rotate-180" : ""}`}
          aria-hidden="true"
        />
      </button>

      {isOpen && (
        <div
          id="auto-explore-database-menu"
          role="dialog"
          aria-labelledby="auto-explore-menu-title"
          className="absolute left-0 top-full z-[60] mt-2 w-72 max-w-[calc(100vw-2rem)] rounded-lg border border-gray-200 bg-white p-4 shadow-xl"
        >
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3
                id="auto-explore-menu-title"
                className="text-sm font-semibold text-gray-900"
              >
                Auto-Explore
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                Choose the academic database to use
              </p>
            </div>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="rounded p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#64748b]"
              aria-label="Close Auto-Explore menu"
            >
              <XMarkIcon className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>

          <div
            className="mt-3 space-y-2"
            role="group"
            aria-label="Academic databases"
          >
            {databases.map((database) => {
              const isSelected = selectedDatabases.includes(database.id);
              return (
                <button
                  key={database.id}
                  type="button"
                  role="checkbox"
                  aria-checked={isSelected}
                  onClick={() => onToggleDatabase(database.id)}
                  className={`flex w-full items-center justify-between rounded-md border px-3 py-2.5 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#64748b] ${
                    isSelected
                      ? "border-[#232F3E] bg-gray-100 text-[#232F3E]"
                      : "border-gray-200 bg-white text-gray-800 hover:border-gray-400 hover:bg-gray-50"
                  }`}
                >
                  <span className="font-medium">{database.label}</span>
                  {isSelected && (
                    <CheckIcon
                      className="h-4 w-4 text-[#232F3E]"
                      aria-hidden="true"
                    />
                  )}
                </button>
              );
            })}
          </div>

          <button
            type="button"
            disabled={selectedDatabases.length === 0 || isSearching || disabled}
            onClick={() => {
              setIsOpen(false);
              onSearch();
            }}
            className="mt-4 w-full rounded-md bg-[#232F3E] px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-[#1A2330] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#64748b] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isSearching ? "Searching…" : "Search"}
          </button>
        </div>
      )}
    </div>
  );
}
