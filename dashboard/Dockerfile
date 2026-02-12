# ---- Build stage ----
FROM node:20-alpine AS builder
WORKDIR /app/dashboard

COPY package.json package-lock.json ./
COPY packages/shared/package.json packages/shared/
COPY packages/server/package.json packages/server/
COPY packages/client/package.json packages/client/
RUN npm ci

COPY tsconfig.base.json ./
COPY packages/shared/ packages/shared/
COPY packages/server/ packages/server/
COPY packages/client/ packages/client/
RUN npm run build -w packages/shared && \
    npm run build -w packages/server && \
    npm run build -w packages/client

# ---- Production stage ----
FROM node:20-alpine
WORKDIR /app/dashboard

COPY package.json package-lock.json ./
COPY packages/shared/package.json packages/shared/
COPY packages/server/package.json packages/server/
RUN npm ci --omit=dev -w packages/server -w packages/shared

# Compiled artifacts
COPY --from=builder /app/dashboard/packages/shared/dist packages/shared/dist
COPY --from=builder /app/dashboard/packages/server/dist packages/server/dist
COPY --from=builder /app/dashboard/packages/client/dist packages/client/dist

ENV NODE_ENV=production
ENV PORT=4400
ENV HOST=0.0.0.0

EXPOSE 4400

CMD ["node", "packages/server/dist/index.js"]
