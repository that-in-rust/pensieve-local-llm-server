# User Journey for Pensieve Local LLM Server


- User runs the following command
    ```bash
    pensieve-local-llm-server -- model-filepath-filename <model-filepath-filename> -- ANTHROPIC_BASE_URL <local-server-address-port> --ANTHROPIC_AUTH_TOKEN <anthropic-auth-token>
    ```
    - This pensieve-local-llm-server runs in the background
    - User goes to Claude Code and enters the following environment variables as above
    - User can thus use Claude Code as if it was running on the cloud - although they are running a local LLM