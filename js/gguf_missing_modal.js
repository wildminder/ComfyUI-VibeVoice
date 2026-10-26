// VibeVoice: notice + one-click install for the optional 'gguf' python package.
//
// The backend emits the 'vibevoice.gguf_missing' server event (modules/notify.py)
// when a user picks a .gguf weight while the package is absent. The frontend drops
// any custom event type that no listener has registered (api-Cwoz8_OZ.js:
// `_registered.has(t.type) ? dispatch : console.error("Unknown message type ...")`),
// so registering the type below is what makes the notice reach the browser at all.
//
// No imports: this file is served as a bare extension module (WEB_DIRECTORY = "./js")
// and reads window.comfyAPI directly (the KJNodes pattern). Every access is
// optional-chained so a headless / no-frontend load is a silent no-op.

// Resolved at module load (core bundles finish before custom-node files are imported,
// see the frontend's loadExtensions). Re-read on use so a late-arriving
// window.comfyAPI still works, and a missing one is a silent no-op.
const { api } = window.comfyAPI?.api ?? {};

function comfyApi() {
    return window.comfyAPI?.api?.api ?? api;
}

const GGUF_MISSING_EVENT = "vibevoice.gguf_missing";
const GGUF_PIP_PACKAGE = "gguf";
const MANAGER_INSTALL_URL = "/customnode/install/pip";

const DIALOG_TITLE = "VibeVoice: missing package";
const ABORT_ACTION = { text: "Abort", value: "abort" };
const INSTALL_ACTION = { text: "Install missing package", value: "install" };
const DISMISS_ACTION = { text: "Close", value: "close" };

// Manager needs BOTH allow_pip_install = true AND a loopback --listen, and reads
// both once at startup — copied from ComfyUI-Manager js/common.js handle403Response.
const MANAGER_403_SECURITY_MESSAGE =
    "ComfyUI-Manager rejected the install. Installing packages from a custom node needs BOTH: " +
    "'allow_pip_install = true' in config.ini ([default] section) AND a loopback --listen " +
    "(127.0.0.1 or ::1). Both are read once when ComfyUI starts, so changing either one needs a " +
    "restart with the server stopped.";

function fallbackMessage(detail) {
    // The canonical wording lives in modules/gguf_quant.py; only used if the payload
    // arrived incomplete. Names the package either way.
    return (
        detail?.message ||
        `Loading .gguf weights requires the '${GGUF_PIP_PACKAGE}' Python package, which is not installed.`
    );
}

function fallbackInstallCmd(detail) {
    // install_cmd is built backend-side from sys.executable, so it targets ComfyUI's own env.
    return detail?.install_cmd || "python -m pip install gguf";
}

function installCommandMessage(detail) {
    return (
        "Install it in ComfyUI's own python environment, then restart ComfyUI to apply:\n\n" +
        `${fallbackInstallCmd(detail)}\n\n` +
        "(a terminal outside ComfyUI works too — just make sure it is the interpreter ComfyUI runs on)"
    );
}

async function showDialog(message, actions) {
    const { ComfyAsyncDialog } = window.comfyAPI?.asyncDialog ?? {};
    if (!ComfyAsyncDialog?.prompt) {
        // No dialog primitive available (headless, or a frontend build without it).
        console.warn(`${DIALOG_TITLE}: ${message}`);
        return null;
    }
    return ComfyAsyncDialog.prompt({ title: DIALOG_TITLE, message, actions });
}

async function installViaManager(detail) {
    let response = null;
    try {
        response = await comfyApi()?.fetchApi(MANAGER_INSTALL_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ packages: GGUF_PIP_PACKAGE }),
        });
    } catch (error) {
        // Network/abort failure: fall through to the copyable-command fallback below.
        console.warn(`${DIALOG_TITLE}: Manager install request failed`, error);
        response = null;
    }

    if (response?.status === 200) {
        await showDialog(
            "Install started in ComfyUI's own environment. Restart ComfyUI to apply.",
            [DISMISS_ACTION]
        );
        return;
    }

    if (response?.status === 403) {
        await showDialog(
            `${MANAGER_403_SECURITY_MESSAGE}\n\n${installCommandMessage(detail)}`,
            [DISMISS_ACTION]
        );
        return;
    }

    // 404, 5xx, or a fetch failure: Manager is not answering as expected.
    const reason = response?.status ? `Manager returned HTTP ${response.status}.` : "";
    await showDialog(
        [reason, installCommandMessage(detail)].filter(Boolean).join("\n\n"),
        [DISMISS_ACTION]
    );
}

async function handleGgufMissing(e) {
    const detail = e?.detail ?? {};
    const choice = await showDialog(fallbackMessage(detail), [ABORT_ACTION, INSTALL_ACTION]);

    if (choice !== "install") {
        // 'abort', an Esc/X dismissal (null), or a missing dialog primitive.
        return;
    }

    if (detail.manager_available) {
        await installViaManager(detail);
        return;
    }

    await showDialog(installCommandMessage(detail), [DISMISS_ACTION]);
}

// Registering the type is what keeps the server event from being dropped.
api?.addEventListener(GGUF_MISSING_EVENT, handleGgufMissing);
