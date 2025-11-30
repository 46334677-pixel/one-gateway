# 使用域名 + Nginx + Certbot 正式上线（可选）

适合：你有自己的域名，想用 `https://你的域名` 对外提供服务。

前提：
- 已在云主机运行 `uvicorn app.main:app`（监听 127.0.0.1:8000 或 0.0.0.0:8000）
- 云主机安全组/防火墙放行 80/443 端口

## 1. 在阿里云购买/管理域名并解析到云主机

1) 登录阿里云控制台 → 域名与网站（域名与 DNS）
2) 购买/已有一个域名，例如 `yourdomain.com`
3) 进入域名的“解析设置”（阿里云 DNS）
   - 新增一条 A 记录：
     - 主机记录：`@`（或你想用的二级如 `api`）
     - 记录类型：A
     - 记录值：你的云主机公网 IP（例如 47.110.92.74）
     - TTL：默认
   - 生效通常需要几分钟到 30 分钟

4) 在云主机上确认解析：
```bash
dig +short yourdomain.com
# 或
ping yourdomain.com -c 1
```

## 2. 安装 Nginx 并配置反向代理

```bash
sudo apt update && sudo apt install -y nginx
```

将 `ops/nginx-onegateway.conf` 上传到 `/etc/nginx/sites-available/onegateway.conf`：
```bash
sudo cp one-gateway/ops/nginx-onegateway.conf /etc/nginx/sites-available/onegateway.conf
sudo ln -sf /etc/nginx/sites-available/onegateway.conf /etc/nginx/sites-enabled/onegateway.conf
sudo nginx -t && sudo systemctl reload nginx
```

## 3. 使用 Certbot 签发免费证书（Let’s Encrypt）

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
# 如果用二级域名：-d api.yourdomain.com
```

Certbot 会自动修改 Nginx 配置并安装证书；证书会自动续期。

## 4. 在 ChatGPT Actions 中导入

- URL 填 `https://yourdomain.com/openapi.json`
- Authentication：API Key in Header → 名称 `X-Api-Key`，值为 `.env` 中的 `GATEWAY_API_KEY`

## 故障排查

- 80/443 打不开：检查安全组/防火墙是否放行；`sudo ss -ltnp` 看监听端口；`sudo nginx -t` 验证配置
- 证书失败：域名解析没生效、Nginx 未指向该域名或 80 端口被占用
- 回源失败：确认 uvicorn 正在 127.0.0.1:8000 提供服务；`tail -f /var/log/nginx/access.log` 边访问边看

