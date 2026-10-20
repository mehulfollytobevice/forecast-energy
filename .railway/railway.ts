import {
  bucket,
  defineRailway,
  github,
  preserve,
  project,
  ref,
  service,
} from "railway/iac";

const REPO = github("mehulfollytobevice/forecast-energy", { branch: "main" });

const DOCKERFILE = {
  builder: "DOCKERFILE" as const,
  dockerfilePath: "Dockerfile",
};

export default defineRailway((ctx) => {
  const snapshots = bucket("snapshots", { region: "iad" });

  const pipelineEnv = {
    PJM_API_KEY: preserve(),
    OPEN_METEO_API_KEY: preserve(),
    S3_BUCKET: ref(snapshots, "BUCKET"),
    S3_ENDPOINT_URL: ref(snapshots, "ENDPOINT"),
    AWS_ACCESS_KEY_ID: ref(snapshots, "ACCESS_KEY_ID"),
    AWS_SECRET_ACCESS_KEY: ref(snapshots, "SECRET_ACCESS_KEY"),
    AWS_DEFAULT_REGION: ref(snapshots, "REGION"),
  };

  const web = service("web", {
    source: REPO,
    build: DOCKERFILE,
    healthcheck: "/api/health",
    healthcheckTimeout: 120,
    deploy: {
      restartPolicyType: "ON_FAILURE",
    },
    env: pipelineEnv,
  });

  const forecast = service("forecast", {
    source: REPO,
    build: DOCKERFILE,
    start: "uv run python -m pjm_forecasting.jobs.forecast",
    deploy: {
      cronSchedule: "15 * * * *",
      restartPolicyType: "NEVER",
    },
    env: pipelineEnv,
  });

  const retrain = service("retrain", {
    source: REPO,
    build: DOCKERFILE,
    start: "uv run python -m pjm_forecasting.jobs.retrain",
    deploy: {
      cronSchedule: "15 6 * * 1",
      restartPolicyType: "NEVER",
    },
    env: pipelineEnv,
  });

  return project(ctx.projectName ?? "forecast-energy", {
    resources: [snapshots, web, forecast, retrain],
  });
});
