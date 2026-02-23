"use client";

type Tab = "viewer" | "alerts" | "timeline" | "devices";

interface SidebarProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

const tabs: { id: Tab; label: string; icon: string }[] = [
  { id: "viewer", label: "3D Viewer", icon: "🏗️" },
  { id: "alerts", label: "Alerts", icon: "🔔" },
  { id: "timeline", label: "Timeline", icon: "📊" },
  { id: "devices", label: "Devices", icon: "📡" },
];

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <aside className="w-16 bg-gray-900 border-r border-gray-800 flex flex-col items-center py-4 shrink-0">
      <div className="w-10 h-10 bg-brand-600 rounded-lg flex items-center justify-center mb-6 text-lg font-bold">
        S
      </div>

      <nav className="flex flex-col gap-2 flex-1">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            title={tab.label}
            className={`w-10 h-10 rounded-lg flex items-center justify-center text-lg transition-colors ${
              activeTab === tab.id
                ? "bg-brand-600/20 text-brand-400"
                : "text-gray-500 hover:text-gray-300 hover:bg-gray-800"
            }`}
          >
            {tab.icon}
          </button>
        ))}
      </nav>
    </aside>
  );
}
