"""Полный тест nightly maintenance."""
import asyncio
import sys
sys.path.insert(0, '/home/blake/memory')

from mcp_server.server import build_context

async def main():
    ctx = await build_context()
    print('✅ Контекст создан')
    
    # 1. Consolidation
    print('\n🔍 1. Consolidation...')
    result = await ctx.consolidator.consolidate_session('default')
    print(f'   Результат: {result}')
    
    # 2. Decay
    print('\n⏳ 2. Decay...')
    decay_result = await ctx.decay.run_night_job()
    print(f'   Результат: {decay_result}')
    
    # 3. Enrichment
    print('\n🎨 3. Enrichment...')
    enrich_result = await ctx.maintenance._enrich_entities(__import__('time').time())
    print(f'   Результат: {enrich_result}')
    
    # 4. Housekeep
    print('\n🧹 4. Housekeep...')
    housekeep_result = await ctx.maintenance._housekeep(__import__('time').time())
    print(f'   Результат: {housekeep_result}')
    
    # 5. Статистика
    print('\n📊 Статистика памяти:')
    stats = await ctx.tools.memory_stats()
    print(f'   nodes_by_type: {stats[\"nodes_by_type\"]}')
    print(f'   entities_by_type: {stats[\"entities_by_type\"]}')
    print(f'   db_size_mb: {stats[\"db_size_mb\"]}')
    
    # Закрываем ресурсы
    await ctx.queue.stop()
    await ctx.embedding.close()
    await ctx.consolidator.close()
    await ctx.db.close()
    print('\n✅ Готово')

asyncio.run(main())
