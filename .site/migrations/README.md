# Content Taxonomy Migration

The Astro build contains fallback redirect pages for the legacy routes. Production redirects should be imported from `content-taxonomy-redirects.csv` into a disabled Cloudflare Bulk Redirect List before cutover.

## Cutover order

1. Deploy the new static site.
2. Verify the new category and representative child routes return `200`.
3. Seed the new page routes into D1 from `.site/workers/ask-docs` with `npm run seed:kb:remote`. The seed removes page-graph rows that no longer exist.
4. Verify `/api/kb/graph` and representative `/api/kb/page?route=...` responses use canonical routes.
5. Enable the Cloudflare Bulk Redirect Rule.
6. Verify every legacy route returns one `301` followed by a `200` response.
7. Verify the sitemap, canonical URLs, search index, `llms.txt`, Open Graph images, and knowledge graph contain only canonical routes.

The CSV follows Cloudflare's headerless Bulk Redirect format:

```text
source,target,status,preserve_query_string,include_subdomains,subpath_matching,preserve_path_suffix
```

Keep the redirects enabled for at least one year. The Astro redirects are a fallback for requests that bypass the Cloudflare custom domain and are not a replacement for edge `301` responses.
