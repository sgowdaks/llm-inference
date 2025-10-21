1. create a config-min.json during export. Store all necessary runtime configs in this. 
2. remove dependency on transformers.
3. use `hf download` command to download the model, and use it for exporting.
4. Add test case for evaluation. Create json file of five examples. That can provide only one correct answer. 
5. Add `hf inference` file and run the above test.