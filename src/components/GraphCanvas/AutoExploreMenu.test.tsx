import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import AutoExploreMenu from "./AutoExploreMenu";
import type { AcademicDatabase } from "../../types/autoExplore";

function MenuHarness({ onSearch }: { onSearch: () => void }) {
  const [selected, setSelected] = useState<AcademicDatabase[]>([]);
  return (
    <AutoExploreMenu
      selectedDatabases={selected}
      onToggleDatabase={(database) =>
        setSelected((current) =>
          current.includes(database)
            ? current.filter((item) => item !== database)
            : [...current, database]
        )
      }
      onSearch={onSearch}
    />
  );
}

describe("AutoExploreMenu", () => {
  it("requires a source and starts a search with the selected source", () => {
    const onSearch = vi.fn();
    render(<MenuHarness onSearch={onSearch} />);

    fireEvent.click(screen.getByRole("button", { name: /auto-explore/i }));
    const searchButton = screen.getByRole("button", { name: "Search" });
    expect(searchButton).toBeDisabled();

    fireEvent.click(screen.getByRole("checkbox", { name: "Arxiv" }));
    expect(searchButton).toBeEnabled();
    fireEvent.click(searchButton);
    expect(onSearch).toHaveBeenCalledOnce();
  });

  it("disables its search action while a search is running", () => {
    render(
      <AutoExploreMenu
        selectedDatabases={["pubmed"]}
        onToggleDatabase={vi.fn()}
        onSearch={vi.fn()}
        isSearching
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /auto-explore/i }));
    expect(screen.getByRole("button", { name: /searching/i })).toBeDisabled();
  });
});
