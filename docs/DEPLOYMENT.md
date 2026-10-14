# Streamlit Community Cloud deployment

**Public demo:** [terramind-demo.streamlit.app](https://terramind-demo.streamlit.app/).

The app is deployed using the cloud entrypoint and CPU-only Linux dependency configuration. Hosted sample inference, the plume overlay, alert JSON with WGS84 coordinates, the Alert Log, and the Model Card were verified on 14 October 2026.

## Deploy

Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and create an app with:

| Setting | Value |
| --- | --- |
| Repository | `prithvi-kaizen/Terramind` |
| Branch | `main` |
| Main file path | `deployment/streamlit_app.py` |
| Python version | `3.12` |
| Secrets | None required |

Choose an available app subdomain in the deployment form. The [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) describes the sign-in, repository selection, and Python settings.

The dependency file beside the entrypoint selects CPU-only PyTorch and torchvision on Linux and includes the pinned root dependencies. Community Cloud checks the entrypoint directory for dependency files before the repository root; see [its dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Runtime

The cloud entrypoint uses direct inference in the Streamlit process. It does not start Flask or call a localhost API. The included checkpoint supplies the encoder and decoder weights, so inference does not download pretrained model files. The full training dataset is unnecessary for the sample demo.

Dashboard prediction files and the alert log are separated by browser session. Uploads are limited to 32 MiB. Prediction files live on the app server and are not durable storage; expect them to disappear after a restart or redeployment.

The shared host supplies CPU inference. The repository's `cloud-smoke` CI job installs the full CPU stack and runs the included sample through this exact entrypoint. [The deployment preparation run](https://github.com/prithvi-kaizen/Terramind/actions/runs/38558347510) passed both the unit-test and full cloud-smoke jobs. Deploying or managing the app requires an authenticated Community Cloud account.

## Verify the deployed app

After the deployment build finishes, select **Sample tile** and click **Run Inference**. Confirm that the plume overlay appears, the alert contains WGS84 coordinates, and the Alert Log and Model Card tabs load. The recorded sample prediction has 1,244 foreground pixels at threshold 0.5; this check is not a new quality benchmark.

Only add the URL to the repository homepage and README after verifying the hosted app. Check the build logs for dependency failures or memory errors rather than assuming a successful deployment from the configuration alone.
