import type { NextApiRequest, NextApiResponse } from 'next'

export default async function handler(
    req: NextApiRequest,
    res: NextApiResponse
) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' })
    }

    try {
        const backendUrl = process.env.NODE_ENV === 'production'
            ? 'https://intelliproofbackend.vercel.app/api/ai/critique-graph'
            : 'http://localhost:8000/api/ai/critique-graph'

        const response = await fetch(backendUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(req.body),
            signal: AbortSignal.timeout(120000),
        })

        const responseText = await response.text()
        let responseData: unknown

        try {
            responseData = JSON.parse(responseText)
        } catch {
            responseData = { details: responseText }
        }

        if (!response.ok) {
            console.error('Critique backend error:', response.status, responseText)
            return res.status(response.status).json({
                error: `Backend error: ${response.status}`,
                ...(typeof responseData === 'object' && responseData !== null
                    ? responseData
                    : { details: responseText }),
            })
        }

        return res.status(200).json(responseData)
    } catch (error) {
        console.error('Critique API route error:', error)
        return res.status(500).json({
            error: 'Internal server error',
            details: error instanceof Error ? error.message : 'Unknown error',
        })
    }
}
