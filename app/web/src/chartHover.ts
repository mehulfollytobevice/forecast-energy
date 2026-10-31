import { useCallback, useState, type PointerEvent, type RefObject } from "react";

export function usePlotHover(
  svgRef: RefObject<SVGSVGElement | null>,
  width: number,
  marginLeft: number,
  plotWidth: number,
  count: number,
) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const onPointerMove = useCallback(
    (event: PointerEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg || count < 2 || plotWidth <= 0) {
        setHoverIndex(null);
        return;
      }
      const rect = svg.getBoundingClientRect();
      if (rect.width <= 0) {
        setHoverIndex(null);
        return;
      }
      const svgX = ((event.clientX - rect.left) / rect.width) * width;
      const t = (svgX - marginLeft) / plotWidth;
      if (t < 0 || t > 1) {
        setHoverIndex(null);
        return;
      }
      setHoverIndex(Math.round(t * (count - 1)));
    },
    [count, marginLeft, plotWidth, svgRef, width],
  );

  const onPointerLeave = useCallback(() => {
    setHoverIndex(null);
  }, []);

  const activeIndex =
    hoverIndex !== null && hoverIndex < count ? hoverIndex : null;

  return { hoverIndex: activeIndex, onPointerMove, onPointerLeave };
}
