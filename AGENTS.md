## Development

The site lives in `packages/web`. When starting the dev server, run from there in background mode:

```
astro dev --background
```

Manage the background server with `astro dev stop`, `astro dev status`, and `astro dev logs`.

## Releases

Publish `runonweb` from `packages/runonweb`, never the private root package. Releases are manual; there is no release workflow.

Ask for explicit confirmation before pushing, publishing to npm, or creating the GitHub release. These steps affect remote services, and pushing `main` also deploys the site through Cloudflare Workers Builds. Never run them just to test the release process.

1. Make sure every requested change is committed, the working tree is clean, and the current branch is `main`. Check the published version and pending commits:

   ```
   git fetch origin
   git status --short --branch
   npm view runonweb version
   git log origin/main..main --oneline
   ```

   Do not release when `main` is behind or has diverged from `origin/main`; integrate the remote changes first.

2. Choose exactly one version bump: `patch` for documentation and compatible fixes, `minor` for compatible features, or `major` for breaking changes.

3. Run the validation from the repository root. Inspect the package contents and stop if anything fails:

   ```
   pnpm check
   pnpm build
   cd packages/runonweb
   npm pack --dry-run
   npm whoami
   gh auth status
   ```

4. After confirmation, run the following from `packages/runonweb` in the same shell. Replace `patch` with the bump selected above. `npm version` creates the version commit and tag.

   ```
   npm version patch
   VERSION="v$(node -p 'require("./package.json").version')"
   git push origin main
   git push origin "$VERSION"
   npm publish --access public
   gh release create "$VERSION" --title "runonweb $VERSION" --generate-notes --latest
   ```

5. Verify that npm reports the new version and that the GitHub release exists:

   ```
   npm view runonweb version
   gh release view "$VERSION"
   ```

If any step fails, stop and fix the cause before continuing. Do not create another version to retry a failed release. If npm publication succeeded but the GitHub release failed, keep the same version and resume at the GitHub step. Published npm version numbers cannot be reused.

## Documentation

Full documentation: https://docs.astro.build

Consult these guides before working on related tasks:

- [Adding pages, dynamic routes, or middleware](https://docs.astro.build/en/guides/routing/)
- [Working with Astro components](https://docs.astro.build/en/basics/astro-components/)
- [Using React, Vue, Svelte, or other framework components](https://docs.astro.build/en/guides/framework-components/)
- [Adding or managing content](https://docs.astro.build/en/guides/content-collections/)
- [Adding styles or using Tailwind](https://docs.astro.build/en/guides/styling/)
- [Supporting multiple languages](https://docs.astro.build/en/guides/internationalization/)
