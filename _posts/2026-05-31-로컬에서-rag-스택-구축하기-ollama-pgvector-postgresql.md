---
layout: post
title: "로컬에서 RAG 스택 구축하기 — Ollama + pgvector + PostgreSQL"
date: 2026-05-31
tags: [RAG, Ollama, pgvector, PostgreSQL, Docker, AI]
categories: [AI, Backend]
---

AI 기반 서비스를 만들 때 가장 흔한 요구사항 중 하나는 **"내 데이터를 기반으로 AI가 답변하게 만들기"**다. 이게 바로 RAG(Retrieval-Augmented Generation)인데, 외부 API 없이 로컬에서 전체 스택을 구성하는 방법을 정리한다.

## 전체 구조

```
① 저장 시: 텍스트 → nomic-embed-text → 벡터 → pgvector 저장
② 질문 시: 질문 → nomic-embed-text → 벡터 → 유사 문서 검색
③ 답변 시: 검색된 맥락 + 질문 → qwen2.5:14b → 답변 생성
```

두 종류의 모델이 필요하다.

| 모델 | 역할 |
|------|------|
| `nomic-embed-text` | 텍스트 → 벡터 변환 전용 (임베딩 모델) |
| `qwen2.5:14b` | 맥락을 읽고 답변 생성 (LLM) |

임베딩 모델은 질문에 답하는 능력이 없다. "삼성전자 실적"을 넣으면 `[0.23, -0.11, 0.87 ...]` 같은 숫자 배열만 반환한다. 이 숫자 배열이 의미의 좌표계가 되어, **의미가 비슷한 텍스트는 좌표가 가깝게** 저장된다.

## 왜 벡터 변환에 모델이 필요한가

단순 키워드 검색과의 차이:

```
키워드 검색: "주가 상승" 검색 → "주가 상승"이라는 단어가 있는 문서만 반환
벡터 검색:  "주가 상승" 검색 → "주가 올랐다", "증시 반등" 등 의미가 유사한 문서도 반환
```

"주가 올랐다"와 "주가 상승"이 같은 의미임을 판단하려면 수백억 문장을 학습한 모델이 필요하다.

## 환경 구성

### 1. Docker로 PostgreSQL + pgvector 실행

```yaml
# docker-compose.yml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: myapp
      POSTGRES_PASSWORD: myapp_dev
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

```bash
docker compose up -d

# pgvector extension 활성화
psql postgresql://myapp:myapp_dev@localhost:5432/myapp \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 2. Ollama로 모델 설치

```bash
# LLM (답변 생성용)
ollama pull qwen2.5:14b

# 임베딩 모델 (벡터 변환용, 270MB)
ollama pull nomic-embed-text
```

### 3. 테이블 스키마

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(768),  -- nomic-embed-text 차원 수
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 유사도 검색 인덱스
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

## Python 구현

```python
import ollama
import psycopg2

conn = psycopg2.connect("postgresql://myapp:myapp_dev@localhost:5432/myapp")

def embed(text: str) -> list[float]:
    res = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return res["embedding"]

def store(text: str):
    vec = embed(text)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (text, vec)
        )
    conn.commit()

def search(query: str, top_k: int = 3) -> list[str]:
    vec = embed(query)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vec, top_k))
        return [row[0] for row in cur.fetchall()]

def ask(question: str) -> str:
    # 유사 문서 검색
    context = "\n\n".join(search(question))

    # LLM으로 답변 생성
    res = ollama.chat(
        model="qwen2.5:14b",
        messages=[{
            "role": "user",
            "content": f"다음 맥락을 참고해서 질문에 답해줘.\n\n맥락:\n{context}\n\n질문: {question}"
        }]
    )
    return res["message"]["content"]
```

## 동작 확인

```python
# 데이터 저장
store("삼성전자 3분기 반도체 수요 증가로 실적 호조")
store("SK하이닉스 HBM 공급 확대, AI 서버 수요 급증")
store("오늘 서울 날씨 맑음, 기온 25도")

# 질문
print(ask("반도체 업황이 어때?"))
# → 삼성전자와 SK하이닉스 관련 문서를 참조해서 답변 생성
# → 날씨 문서는 의미가 달라 검색되지 않음
```

## 정리

외부 API 없이 로컬에서 RAG 전체 스택을 구성할 수 있다. `nomic-embed-text`(임베딩) + `qwen2.5:14b`(LLM) + `pgvector`(벡터 DB) 조합으로 개발 단계에서는 비용 없이 구조를 잡고, 나중에 Claude API나 OpenAI로 교체할 수 있도록 추상화해두는 것이 핵심이다.
