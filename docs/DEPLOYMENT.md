# Deployment

The site runs on **Google Cloud Run** and auto-deploys on every push to `main`
via `.github/workflows/google-cloudrun-docker.yml`. The workflow authenticates
keylessly using **Workload Identity Federation (WIF)** — no service-account keys
are stored in GitHub.

You only need to do the one-time setup below once. After that, every merge to
`main` redeploys automatically.

---

## 1. One-time Google Cloud setup

Run these in Cloud Shell (or locally with `gcloud`). Fill in the values at top.

First, confirm your Cloud Run service name and region (the project name is not
the service name):

```bash
gcloud run services list --project robotic-branch-452315-m3
```

Look at the `SERVICE` and `REGION` columns. Based on the live URL
(`idx-test2-3592241-4xeemoth6q-ul.a.run.app`) the service is almost certainly
`idx-test2-3592241` — but use whatever the command prints.

```bash
# ---- fill these in ----
export PROJECT_ID="robotic-branch-452315-m3"
export REGION="us-east5"                       # confirm with the command above
export SERVICE="idx-test2-3592241"             # confirm with the command above
export REPO="ericjnielson/eric-nielson-portfolio"
export DEPLOYER="gh-deployer"                  # name for the deploy service account
# -----------------------

export PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud config set project "$PROJECT_ID"

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com

# Create the deployer service account the Action will impersonate
gcloud iam service-accounts create "$DEPLOYER" \
  --display-name="GitHub Actions deployer"
export SA_EMAIL="${DEPLOYER}@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant it the roles needed for a source-based Cloud Run deploy
for ROLE in \
  roles/run.admin \
  roles/cloudbuild.builds.editor \
  roles/artifactregistry.admin \
  roles/storage.admin \
  roles/iam.serviceAccountUser ; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" --role="$ROLE"
done
```

## 2. Create the Workload Identity pool + provider

```bash
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" --display-name="GitHub pool"

gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" --workload-identity-pool="github-pool" \
  --display-name="GitHub provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${REPO}'"

# Allow the GitHub repo's identity to impersonate the deployer service account
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${REPO}"

# Print the provider resource string you'll paste into GitHub
echo "GCP_WORKLOAD_IDENTITY_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
echo "GCP_DEPLOY_SERVICE_ACCOUNT     = ${SA_EMAIL}"
```

## 3. Add GitHub repository Variables

In GitHub: **Settings → Secrets and variables → Actions → Variables → New
repository variable**. Add:

| Variable | Value |
| --- | --- |
| `GCP_PROJECT_ID` | your project id |
| `CLOUD_RUN_SERVICE` | your Cloud Run service name |
| `GCP_REGION` | `us-east5` (optional; this is the default) |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | the `projects/.../providers/github-provider` string printed above |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | the `gh-deployer@...` email printed above |

These are **Variables**, not Secrets — none of them are sensitive (the WIF
provider string and SA email are safe to expose; security comes from the
attribute condition binding the pool to this repo).

## 4. Deploy

Push to `main`, or run the workflow manually from the **Actions** tab
("Deploy to Cloud Run" → **Run workflow**). The job prints the live URL when it
finishes.

---

## Notes

- **Env vars are preserved.** The workflow does not override env vars, so values
  already set on the service (e.g. `OPENAI_API_KEY`) carry over. To change them,
  edit the service in the Cloud Run console or add an `env_vars` block to the
  deploy step.
- **First run builds from source.** `--source .` uses Cloud Build to build the
  `Dockerfile` and push to an auto-created `cloud-run-source-deploy` Artifact
  Registry repo. If a build fails with a permission error, grant the named role
  to the deployer SA and re-run.
- **Manual deploy** (no Action) remains available:
  `gcloud run deploy "$SERVICE" --source . --region "$REGION"`.
