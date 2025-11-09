/tear/dasulimov/home_folder/mass_spec_papers_multi

output 
/home/tk-lpt-0806/Desktop/pdf_to_process/input

05a12eb3b6ce838fd91e1a38aea7e743cec5c941.pdf  0b5c8d99b0dc16f850f3091db5eb7fe802481889.pdf  10fb849f99b350910fc37aa09913ad703031c17d.pdf  6e42beb1595ef78ee1e3733755aa218eccdced52.pdf
05a1646dacc7c3e34781eee08516b09d1980ce8f.pdf  0b5cc3452143a1f121b28847ed26b944351d28ca.pdf  10fc965726c2acd5adfa68171ec4bfccfd0f1bcc.pdf  6e43af6bab1b82479b0a56d811f6299e24238dba.pdf
05a4bdfafa1cfd5fff831f5a214f06c379df9c80.pdf  0b5cde19dc72ab09ccf0ecd222394c13e32ac664.pdf  10fd09cb2ff9b004c9ffa4043509a39313ff4ec1.pdf  6e449ec39938802aab298ef5caf065e2d46e0090.pdf
05a4d9073ab3f7667c1205b6e1b34aaa0fa2cad5.pdf  0b5d022e8f00be10791bd4f7c0b1f49704792958.pdf  10fe8e74e04c5818b63d715713b9ba4f17c42b4e.pdf  6e466a02c60c8576051cd0ef4a6d13503f8e955c.pdf
05a6ecfab52032bdc58bcdb4e1cee5b196e103df.pdf  0b5da707da3a5a8b4c35f897642cf02228444af7.pdf  10fec52f5c72f3317de6dd962ebd4f3c250aaaba.pdf  6e478288ec862f36e68667f641ebf4b66da9723c.pdf
05a7bb789699145ed1a260153c4be0e2eb3631f7.pdf  0b607e2c4a098a3fef82c57b8cfeb6141c2e28b9.pdf  10feddce0e2458008ceec90b34cfbab596655685.pdf  6e47c70393f8f9b5c51803286a59b04b5e99da00.pdf
05a7e56ad2ce37bba9d6a645b654be661700c619.pdf  0b60c009a7cb9bba2b7dc60abaaee49b8fbff728.pdf  110013a72c58fbe1958f7bb89ee4eb41fc579d56.pdf  6e48418aedb2ef18afbb35b2bd736aaf8901f144.pdf
05a84b9ecca325a292e5ff3b222a332752b01c7c.pdf  0b60c2fa8f6414a3295fe41c76d04596825835da.pdf  1100716b189144d8283039ac4bc6f09de55d5165.pdf  6e4f9b49b29c16508ceb52a42c30761cf6c60ec7.pdf
05a909bb7074cbae598ba84407ac3c706e13c9b9.pdf  0b6222be7dd7bcb05119ce3581004220ba4fedd9.pdf  11016dd0f84c85c3e487d1896907238bcac2b269.pdf  6e4fc532e0b24dadba3e6fd2c276ea2b05de800b.pdf
05a960df994806a303c2fca69123fd39b59ca89a.pdf  0b6251b02c4050a45a8adccd938a4339df9e5508.pdf  1101b0e1879d173c4767d1119cf09889a018940d.pdf  6e508485dd56ed306449c2c4ec1d47f761fcc7ca.pdf
05a9b35f85a76c7bd9678b360195656f84cdb0e2.pdf  0b640a67223a89a49a0464320142ede47d59b843.pdf  1101b7471b14d4f8662022aa0f65830b265a02de.pdf  6e50af90e45e718c6612bd8091666605dffab04f.pdf
05aa202ed8600f13f856d88953715863e8775ead.pdf  0b64a5ae29068debf36a4a50ced4a8d2e269faf1.pdf  1101e94002637c7923a63ac8906e6c4f94d911e6.pdf  6e515ca0b8e12f62a68a8e08b827dc8530ff9c64.pdf
05aa731abe539f544d20f25823e2d4af3c85b74d.pdf  0b66b577e4a56cd6616f50fe755ac79aeaccbd9b.pdf  11060845cd9881ee617f7b9c65f2a3e04cac95b0.pdf  6e59d0891f6ff388fba38d7f2e4941588e2ed40c.pdf

Create Sample files 
mkdir -p /home/asad/sample_files && i=0 && find "/tear/dasulimov/home_folder/mass_spec_papers_multi" -type f -print0 | while IFS= read -r -d '' file; do cp -- "$file" "/home/asad/sample_files/"; i=$((i+1)); [ "$i" -ge 1000 ] && break; done



//vision Script
# Use config file (auto-detected)
python vision_extractor.py

nohup python -u src/vision_extractors/vision_extractor.py > logs/vision_extractor.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup ollama serve > ollama-server.log 2>&1 &