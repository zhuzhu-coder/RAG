# Recipe Dataset Expansion Design

## Goal

Expand the local recipe knowledge base from 1 Markdown recipe to 60 Markdown recipes so the RAG project is more credible for resume presentation, retrieval testing, and live demos.

## Scope

- Keep the current loader, metadata extraction, and chunking logic unchanged.
- Add 59 new Markdown files under `data/cook`.
- Keep the existing `data/cook/vegetable_dish/番茄炒蛋.md` file.
- Cover all existing category directory names used by `DataPreparationModule.CATEGORY_MAPPING`.

## Dataset Shape

The final dataset will contain 60 recipes:

- `meat_dish`: 8 recipes
- `vegetable_dish`: 8 recipes including the existing recipe
- `soup`: 7 recipes
- `dessert`: 6 recipes
- `breakfast`: 7 recipes
- `staple`: 7 recipes
- `aquatic`: 6 recipes
- `condiment`: 5 recipes
- `drink`: 6 recipes

Every new file follows the current Markdown structure:

- `# 菜品名称`
- `## 基本信息`
- `## 必备原料`
- `## 操作步骤`
- `## 小贴士`

Difficulty remains encoded with `★` through `★★★★★`, because the existing parser recognizes difficulty from those characters.

## Verification

After adding files:

- Run `python -m pytest tests -q -p no:cacheprovider` from `code/`.
- Count Markdown files under `data/cook` and confirm the total is 60.
- Load and chunk the dataset with `DataPreparationModule` to confirm category statistics and chunk generation work.
