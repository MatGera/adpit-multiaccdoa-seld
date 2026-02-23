FROM node:22-alpine AS builder

WORKDIR /app

# Install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy workspace config
COPY pnpm-workspace.yaml package.json pnpm-lock.yaml* ./
COPY packages/shared-types ./packages/shared-types
COPY frontend/package.json ./frontend/

# Install deps
RUN pnpm install --frozen-lockfile || pnpm install

# Copy source
COPY frontend ./frontend

# Build
WORKDIR /app/frontend
RUN pnpm build

# Production stage
FROM node:22-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/frontend/.next/standalone ./
COPY --from=builder /app/frontend/.next/static ./.next/static
COPY --from=builder /app/frontend/public ./public

EXPOSE 3000

CMD ["node", "server.js"]
