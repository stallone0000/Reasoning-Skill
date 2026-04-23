#!/bin/bash
# Shared shell helpers for reasoning_memory 360 API scripts.

unset_qihoo_proxies() {
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
}

require_qihoo_api_key() {
    API_KEY_VALUE="${API_KEY_360:-${QINIU_API_KEY:-${QIHOO_API_KEY:-${API_KEY:-}}}}"
    if [[ -z "${API_KEY_VALUE}" ]]; then
        echo "ERROR: missing API key env. Set API_KEY_360 or QINIU_API_KEY or QIHOO_API_KEY or API_KEY." >&2
        return 1
    fi

    export API_KEY_VALUE
    export API_KEY_360="${API_KEY_VALUE}"
    export QINIU_API_KEY="${API_KEY_VALUE}"
    export QIHOO_API_KEY="${API_KEY_VALUE}"
    export API_KEY="${API_KEY_VALUE}"
}

