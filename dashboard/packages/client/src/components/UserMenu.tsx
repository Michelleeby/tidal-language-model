import { useState, useRef, useEffect } from "react";
import { useAuthStore, logout } from "../hooks/useAuth.js";
import AllowedUsersPanel from "./AllowedUsersPanel.js";

export default function UserMenu() {
  const user = useAuthStore((s) => s.user);
  const [open, setOpen] = useState(false);
  const [showAccessPanel, setShowAccessPanel] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  if (!user) return null;

  return (
    <div className="relative ml-auto" ref={menuRef}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 rounded-full hover:ring-2 hover:ring-gray-600 transition-all"
      >
        {user.githubAvatarUrl ? (
          <img
            src={user.githubAvatarUrl}
            alt={user.githubLogin}
            className="w-7 h-7 rounded-full"
          />
        ) : (
          <div className="w-7 h-7 rounded-full bg-gray-700 flex items-center justify-center text-xs font-medium text-gray-300">
            {user.githubLogin[0].toUpperCase()}
          </div>
        )}
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-48 bg-gray-900 border border-gray-700 rounded-lg shadow-xl py-1 z-50">
          <div className="px-4 py-2 border-b border-gray-700">
            <p className="text-sm font-medium text-gray-100">
              {user.githubLogin}
            </p>
          </div>
          <button
            onClick={() => {
              setOpen(false);
              setShowAccessPanel(true);
            }}
            className="w-full text-left px-4 py-2 text-sm text-gray-400 hover:text-gray-100 hover:bg-gray-800 transition-colors"
          >
            Manage Access
          </button>
          <button
            onClick={async () => {
              setOpen(false);
              await logout();
            }}
            className="w-full text-left px-4 py-2 text-sm text-gray-400 hover:text-gray-100 hover:bg-gray-800 transition-colors"
          >
            Sign out
          </button>
        </div>
      )}

      {showAccessPanel && (
        <AllowedUsersPanel onClose={() => setShowAccessPanel(false)} />
      )}
    </div>
  );
}
