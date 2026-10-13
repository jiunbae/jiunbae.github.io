# Service favicon snapshots

Projects and My Services use the same catalog icons. These files copy the deployed
service favicons as of 2026-10-13; no logo artwork is generated. Local snapshots avoid
cross-service authentication, changing asset hashes and service downtime breaking icons.
Transparent icons have a light backing in both themes so dark brand marks stay visible.

| Catalog ID | Original source |
| --- | --- |
| munjit | https://munjit.jiun.dev/favicon.ico |
| lol-manager | Inline SVG favicon in https://lol.jiun.dev/ |
| muxa | https://muxa.jiun.dev/static/icon.svg |
| somun | https://somun.jiun.dev/assets/favicon-DINSWbzd.svg |
| sai | https://sai.jiun.dev/favicon.svg |
| bubbles | Inline SVG favicon in https://bubbles.jiun.dev/ |
| daily | https://daily.jiun.dev/favicon.ico |

Update the asset and `src/data/services.ts` together, deploy the blog, then run
`bun run sync:services` in jiun-api and deploy its snapshot. Services without a
published favicon keep the text fallback; a successful HTML response at
`/favicon.ico` is not an image.
