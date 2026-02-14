import type { MetricPoint } from "@tidal/shared";

interface MetricCardsProps {
  latest: MetricPoint | null;
}

export default function MetricCards({ latest }: MetricCardsProps) {
  if (!latest) {
    return null;
  }

  const cards = [
    {
      label: "Loss",
      value: (latest["Losses/Total"] as number)?.toFixed(4) ?? "—",
      color: "text-blue-400",
    },
    {
      label: "Learning Rate",
      value: (latest["Learning Rate"] as number)?.toExponential(2) ?? "—",
      color: "text-emerald-400",
    },
    {
      label: "Iterations/s",
      value:
        (latest["Iterations/Second"] as number)?.toFixed(1) ?? "—",
      color: "text-amber-400",
    },
    {
      label: "Perplexity",
      value: Math.exp((latest["Losses/Total"] as number) ?? 0).toFixed(1),
      color: "text-purple-400",
    },
  ];

  return (
    <>
      {cards.map((card) => (
        <div
          key={card.label}
          className="snap-center flex-shrink-0 w-full md:w-auto bg-gray-900 rounded-lg p-3"
        >
          <div className="text-xs text-gray-500 mb-1">{card.label}</div>
          <div className={`text-2xl font-mono font-semibold ${card.color}`}>
            {card.value}
          </div>
        </div>
      ))}
    </>
  );
}
