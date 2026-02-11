import { useRef, useEffect, useState, type ReactNode } from "react";

interface MetricCarouselProps {
  children: ReactNode;
}

export default function MetricCarousel({ children }: MetricCarouselProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [dotCount, setDotCount] = useState(0);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;

    const updateDotCount = () => {
      setDotCount(container.children.length);
    };
    updateDotCount();

    const observer = new MutationObserver(updateDotCount);
    observer.observe(container, { childList: true });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;

    const onScroll = () => {
      const center = container.scrollLeft + container.clientWidth / 2;
      let closest = 0;
      let minDist = Infinity;
      for (let i = 0; i < container.children.length; i++) {
        const child = container.children[i] as HTMLElement;
        const childCenter = child.offsetLeft + child.offsetWidth / 2;
        const dist = Math.abs(center - childCenter);
        if (dist < minDist) {
          minDist = dist;
          closest = i;
        }
      }
      setActiveIndex(closest);
    };

    container.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => container.removeEventListener("scroll", onScroll);
  }, []);

  const scrollTo = (index: number) => {
    const container = scrollRef.current;
    if (!container) return;
    const child = container.children[index] as HTMLElement | undefined;
    child?.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
  };

  return (
    <div>
      <div
        ref={scrollRef}
        className="flex gap-3 overflow-x-auto snap-x snap-mandatory scrollbar-hide md:grid md:grid-cols-5 md:overflow-x-visible md:snap-none"
      >
        {children}
      </div>

      {/* Dot indicators — mobile only */}
      {dotCount > 1 && (
        <div className="flex justify-center gap-2 mt-3 md:hidden">
          {Array.from({ length: dotCount }, (_, i) => (
            <button
              key={i}
              onClick={() => scrollTo(i)}
              aria-label={`Go to card ${i + 1}`}
              className={`w-2 h-2 rounded-full transition-colors ${
                i === activeIndex ? "bg-blue-400" : "bg-gray-600"
              }`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
