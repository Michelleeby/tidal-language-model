# ---- Build stage ----
FROM node:20-alpine AS builder
WORKDIR /app/dashboard

COPY dashboard/package.json dashboard/package-lock.json ./
COPY dashboard/packages/shared/package.json packages/shared/
COPY dashboard/packages/server/package.json packages/server/
COPY dashboard/packages/client/package.json packages/client/
RUN npm ci

COPY dashboard/tsconfig.base.json ./
COPY dashboard/packages/shared/ packages/shared/
COPY dashboard/packages/server/ packages/server/
COPY dashboard/packages/client/ packages/client/
RUN npm run build -w packages/shared && \
    npm run build -w packages/server && \
    npm run build -w packages/client

# ---- Production stage ----
FROM node:20-alpine
RUN apk add --no-cache git
WORKDIR /app/dashboard

COPY dashboard/package.json dashboard/package-lock.json ./
COPY dashboard/packages/shared/package.json packages/shared/
COPY dashboard/packages/server/package.json packages/server/
RUN npm ci --omit=dev -w packages/server -w packages/shared

# Compiled artifacts
COPY --from=builder /app/dashboard/packages/shared/dist packages/shared/dist
COPY --from=builder /app/dashboard/packages/server/dist packages/server/dist
COPY --from=builder /app/dashboard/packages/client/dist packages/client/dist

# Plugin manifests (projectRoot resolves to /app, registry looks in /app/plugins/)
COPY plugins/ /app/plugins/

ENV NODE_ENV=production
ENV PORT=4400
ENV HOST=0.0.0.0

EXPOSE 4400

CMD ["node", "packages/server/dist/index.js"]
