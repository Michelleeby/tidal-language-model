interface ZoomResetButtonProps {
  visible: boolean;
  onReset: () => void;
}

export default function ZoomResetButton({ visible, onReset }: ZoomResetButtonProps) {
  if (!visible) return null;

  return (
    <button
      onClick={onReset}
      className="absolute top-2 right-2 z-10 px-2 py-1 text-xs font-medium text-gray-300 bg-gray-800 border border-gray-600 rounded hover:bg-gray-700 hover:text-white transition-colors"
    >
      Reset Zoom
    </button>
  );
}
