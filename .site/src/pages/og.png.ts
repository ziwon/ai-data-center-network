import { generateOpenGraphImage } from 'astro-og-canvas';

export const prerender = true;

export async function GET() {
  const image = await generateOpenGraphImage({
    title: 'AI Data Center Systems',
    description: 'Networking, inference, training, MLOps, storage, and performance engineering study notes.',
    bgGradient: [
      [2, 8, 23],
      [3, 21, 45],
      [17, 17, 17],
    ],
    border: {
      color: [217, 57, 46],
      width: 14,
      side: 'inline-start',
    },
    padding: 72,
    font: {
      title: {
        color: [248, 244, 237],
        size: 76,
        weight: 'Bold',
        lineHeight: 0.96,
      },
      description: {
        color: [216, 209, 199],
        size: 34,
        lineHeight: 1.24,
      },
    },
  });

  return new Response(image, {
    headers: {
      'Content-Type': 'image/png',
      'Cache-Control': 'public, max-age=31536000, immutable',
    },
  });
}
