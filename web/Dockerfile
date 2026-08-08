# Multi-stage so the runtime image carries the build output and production
# dependencies only, not the toolchain.
FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY --from=builder /app/.next ./.next
# landing.html is read off disk at request time by app/route.ts, so it has to
# be in the runtime image — it is not bundled into .next.
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.ts ./

# node:alpine already ships an unprivileged `node` user (uid 1000). Use it
# rather than minting another — same reasoning as the API image: nothing here
# needs to write to its own source tree at runtime.
USER node

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD node -e "require('http').get('http://127.0.0.1:3000/',r=>process.exit(r.statusCode<500?0:1)).on('error',()=>process.exit(1))"

CMD ["npm", "run", "start"]
